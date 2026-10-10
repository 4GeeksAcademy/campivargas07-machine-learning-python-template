import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app import INPUT, OUTPUT, infer_reviews, prepare_reviews, read_reviews, select_review_sample, sentiment_summary, star_to_sentiment, validate_results, verify_saved_csv


class SentimentContractTest(unittest.TestCase):
    def test_five_star_mapping(self):
        self.assertEqual([star_to_sentiment(stars) for stars in range(1, 6)],
                         ["negativo", "negativo", "neutral", "positivo", "positivo"])
        for invalid in [0, 6, True, "5", None]:
            with self.assertRaises(ValueError):
                star_to_sentiment(invalid)

    def test_preserve_rows_duplicates_and_originals(self):
        original = pd.DataFrame({"id": ["001", "002", "003", "004"],
                                 "text": [" No esta mal! ", "", " No esta mal! ", None],
                                 "rating": ["5", "", "4", "invalid"]})
        prepared, audit = prepare_reviews(original, language_detector=lambda text: "indeterminado")

        def tokenizer(texts, **kwargs):
            return {"input_ids": [[1, 2, 3, 4] for text in texts]}

        def classifier(texts, **kwargs):
            return [{"label": "4 stars", "score": 0.8} for text in texts]

        result = infer_reviews(prepared, classifier, tokenizer, limit=3, batch_size=1)
        validate_results(original, result)
        self.assertEqual(len(result), 4)
        self.assertEqual(audit["utilizables"], 2)
        self.assertEqual(audit["duplicados_texto_adicionales"], 1)
        self.assertEqual(result.wl_truncada.tolist(), [True, False, True, False])
        self.assertEqual(result.wl_texto_limpio.iloc[0], "No esta mal!")
        self.assertEqual(sentiment_summary(result).conteo.tolist(), [0, 0, 2])
        self.assertAlmostEqual(sentiment_summary(result).porcentaje.sum(), 100)
        broken = result.copy()
        broken.loc[0, "wl_estrellas"] = pd.NA
        with self.assertRaises(AssertionError):
            validate_results(original, broken)

    def test_csv_round_trip_and_reproducible_selection(self):
        original = pd.DataFrame({"id": ["001", "002", "003"],
                                 "text": ['NA', 'Texto, con "comillas"\ny salto!', ''],
                                 "rating": ['5', '', '0']})
        prepared, audit = prepare_reviews(original, language_detector=lambda text: "indeterminado")
        tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2] for text in texts]}
        classifier = lambda texts, **kwargs: [{"label": "3 stars", "score": 0.333333333} for text in texts]
        result = infer_reviews(prepared, classifier, tokenizer, 512)
        validate_results(original, result)
        self.assertEqual(audit['ratings_no_vacios_invalidos'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.csv'
            result.to_csv(path, index=False)
            verify_saved_csv(result, path)
            self.assertEqual(read_reviews(path).id.iloc[0], '001')
            self.assertEqual(read_reviews(path).text.iloc[0], 'NA')
        first = select_review_sample(result)
        second = select_review_sample(result)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(first.wl_fila.is_unique)
        extended = pd.concat([result] * 8, ignore_index=True)
        extended['wl_fila'] = range(len(extended))
        extended_sample = select_review_sample(extended)
        self.assertEqual(len(extended_sample), 16)
        self.assertTrue(extended_sample.wl_fila.is_unique)
        self.assertEqual(extended_sample.motivo_seleccion.eq('aleatoria (semilla 42)').sum(), 10)

    def test_ambiguous_columns_and_collisions_fail_explicitly(self):
        with self.assertRaises(ValueError):
            prepare_reviews(pd.DataFrame({'text': [], 'review': []}))
        with self.assertRaises(ValueError):
            prepare_reviews(pd.DataFrame({'text': [], 'wl_fila': []}))

    def test_duplicate_csv_headers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.csv'
            path.write_text('text,text\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_reviews(path)

    @unittest.skipUnless(INPUT.is_file() and OUTPUT.is_file(), 'Se necesitan CSV reales de entrada y salida.')
    def test_real_csv_coverage_and_original_preservation(self):
        original = read_reviews(INPUT)
        saved = read_reviews(OUTPUT)
        pd.testing.assert_frame_equal(original, saved[list(original.columns)])
        self.assertEqual(len(original), 500)
        self.assertEqual(saved.wl_fila.astype(int).tolist(), list(range(500)))
        self.assertTrue(saved.wl_utilizable.eq('True').all())
        stars = pd.to_numeric(saved.wl_estrellas, errors='raise')
        self.assertTrue(stars.isin([1, 2, 3, 4, 5]).all())
        self.assertEqual(stars.map(star_to_sentiment).tolist(), saved.wl_sentimiento.tolist())
        scores = pd.to_numeric(saved.wl_score, errors='raise')
        self.assertTrue(scores.between(0, 1).all())
        percentages = saved.wl_sentimiento.value_counts().reindex(['negativo', 'neutral', 'positivo'], fill_value=0) / 500 * 100
        self.assertAlmostEqual(percentages.sum(), 100)


if __name__ == "__main__":
    unittest.main()