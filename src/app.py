"""Inferencia reproducible de sentimiento para WeLoveReviews, sin entrenamiento."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/reviews.csv"
OUTPUT = ROOT / "data/processed/reviews_with_sentiment.csv"
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
BANDS = ["negativo", "neutral", "positivo"]
SUPPORTED_LANGUAGES = {"en", "nl", "de", "fr", "it", "es"}
ADDED_COLUMNS = [
	"wl_fila", "wl_texto_limpio", "wl_utilizable", "wl_motivo",
	"wl_idioma", "wl_estrellas_humanas", "wl_tokens", "wl_truncada",
	"wl_estrellas", "wl_sentimiento", "wl_score", "wl_modelo", "wl_revision",
]


def star_to_sentiment(stars):
	if isinstance(stars, bool) or stars not in (1, 2, 3, 4, 5):
		raise ValueError("Las estrellas deben ser un entero entre 1 y 5.")
	return "negativo" if stars <= 2 else "neutral" if stars == 3 else "positivo"


def clean_text(value):
	return " ".join(value.split()) if isinstance(value, str) else ""


def resolve_path(path):
	path = Path(path).expanduser()
	return path if path.is_absolute() else ROOT / path


def read_reviews(path=INPUT, separator=",", encoding="utf-8-sig"):
	path = resolve_path(path)
	if not path.is_file():
		raise FileNotFoundError(
			f"Falta {path}. Coloca alli el CSV real de la plataforma; "
			"no se sustituira por otros datos."
		)
	if len(separator) != 1:
		raise ValueError("El separador debe ser un solo caracter.")
	with path.open(encoding=encoding, newline="") as stream:
		header = next(csv.reader(stream, delimiter=separator), [])
	if not header or len(header) != len(set(header)):
		raise ValueError("La cabecera esta vacia o contiene nombres duplicados.")
	return pd.read_csv(path, sep=separator, encoding=encoding, dtype=str,
					   keep_default_na=False, skip_blank_lines=False)


def choose_column(frame, explicit, aliases, required=False):
	if explicit is not None:
		if explicit not in frame.columns:
			raise ValueError(f"Columna inexistente: {explicit}. Disponibles: {list(frame.columns)}")
		return explicit
	matches = [column for column in frame.columns if column.strip().casefold() in aliases]
	if len(matches) > 1 or (required and not matches):
		raise ValueError(
			f"Selecciona la columna explicitamente; candidatas: {matches}; "
			f"disponibles: {list(frame.columns)}."
		)
	return matches[0] if matches else None


def detect_language(text):
	from langdetect import DetectorFactory, detect
	from langdetect.lang_detect_exception import LangDetectException

	DetectorFactory.seed = 42
	if len(re.findall(r"[^\W\d_]", text, re.UNICODE)) < 20:
		return "indeterminado"
	try:
		return detect(text)
	except LangDetectException:
		return "indeterminado"


def prepare_reviews(frame, text_column=None, rating_column=None, id_column=None,
					language_detector=detect_language):
	if not frame.columns.is_unique:
		raise ValueError("Hay nombres de columnas duplicados; resuelvelos antes del analisis.")
	collisions = set(frame.columns) & set(ADDED_COLUMNS)
	if collisions:
		raise ValueError(f"Las columnas originales colisionan con la salida: {sorted(collisions)}")
	text_column = choose_column(frame, text_column,
							   {"text", "review", "reviews", "review_text", "reviewtext",
								"review text", "texto", "reseña", "resena", "comentario"}, True)
	rating_column = choose_column(frame, rating_column,
								 {"rating", "stars", "score", "estrellas", "puntuacion", "puntuación"})
	id_column = choose_column(frame, id_column, {"id", "review_id", "reviewid", "identificador"})
	result = frame.copy(deep=True).reset_index(drop=True)
	result["wl_fila"] = range(len(result))
	result["wl_texto_limpio"] = result[text_column].map(clean_text)
	result["wl_utilizable"] = result["wl_texto_limpio"].ne("")
	result["wl_motivo"] = result["wl_utilizable"].map({True: "", False: "texto ausente, vacio o no textual"})
	result["wl_idioma"] = result["wl_texto_limpio"].map(language_detector)
	result["wl_estrellas_humanas"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
	if rating_column:
		numeric = pd.to_numeric(result[rating_column], errors="coerce")
		valid = numeric.isin([1, 2, 3, 4, 5])
		result.loc[valid, "wl_estrellas_humanas"] = numeric[valid]
	result["wl_tokens"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
	result["wl_truncada"] = False
	result["wl_estrellas"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
	result["wl_sentimiento"] = pd.Series(pd.NA, index=result.index, dtype="string")
	result["wl_score"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
	result["wl_modelo"] = MODEL_ID
	result["wl_revision"] = MODEL_REVISION
	audit = {
		"filas": len(result), "utilizables": int(result.wl_utilizable.sum()),
		"no_utilizables": int((~result.wl_utilizable).sum()),
		"diferencia_frente_a_500": int(result.wl_utilizable.sum()) - 500,
		"columna_texto": text_column, "columna_rating": rating_column, "columna_id": id_column,
		"nulos_texto": int(frame[text_column].isna().sum()),
		"vacios_texto": int(result.wl_texto_limpio.eq("").sum()),
		"duplicados_filas_adicionales": int(frame.duplicated().sum()),
		"duplicados_texto_adicionales": int(result.loc[result.wl_utilizable, text_column].duplicated().sum()),
		"duplicados_id_adicionales": int(result.loc[result[id_column].notna() & result[id_column].ne(""), id_column].duplicated().sum()) if id_column else None,
		"ratings_validos": int(result.wl_estrellas_humanas.notna().sum()),
		"ratings_no_vacios_invalidos": int((result[rating_column].fillna("").ne("") & result.wl_estrellas_humanas.isna()).sum()) if rating_column else 0,
		"idiomas_estimados": result.loc[result.wl_utilizable, "wl_idioma"].value_counts().to_dict(),
	}
	return result, audit


def load_classifier(device="cpu"):
	from huggingface_hub.constants import HF_HUB_CACHE
	from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

	if Path(HF_HUB_CACHE).expanduser().resolve().is_relative_to(ROOT):
		raise ValueError("La cache de Hugging Face debe estar fuera del repositorio.")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
	)
	model.eval()
	expected = {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}
	if set(model.config.id2label.values()) != expected:
		raise ValueError("Las etiquetas del modelo no coinciden con el mapeo verificado.")
	limit = min(tokenizer.model_max_length, model.config.max_position_embeddings)
	tokenizer.truncation_side = "right"
	classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
	return classifier, tokenizer, int(limit)


def infer_reviews(prepared, classifier, tokenizer, limit, batch_size=8):
	if batch_size < 1:
		raise ValueError("batch_size debe ser positivo.")
	result = prepared.copy(deep=True)
	indices = result.index[result.wl_utilizable].tolist()
	for start in range(0, len(indices), batch_size):
		positions = indices[start:start + batch_size]
		texts = result.loc[positions, "wl_texto_limpio"].tolist()
		encoded = tokenizer(texts, truncation=False, add_special_tokens=True)
		lengths = [len(tokens) for tokens in encoded["input_ids"]]
		predictions = classifier(texts, batch_size=batch_size, truncation=True, max_length=limit)
		if len(predictions) != len(positions):
			raise ValueError("El modelo no devolvio una prediccion por texto.")
		for position, length, prediction in zip(positions, lengths, predictions):
			match = re.fullmatch(r"([1-5]) stars?", prediction["label"])
			if not match:
				raise ValueError(f"Etiqueta desconocida: {prediction['label']}")
			stars = int(match.group(1))
			result.loc[position, ["wl_tokens", "wl_truncada", "wl_estrellas", "wl_sentimiento", "wl_score"]] = [
				length, length > limit, stars, star_to_sentiment(stars), float(prediction["score"])
			]
	return result


def sentiment_summary(result):
	counts = result.wl_sentimiento.value_counts().reindex(BANDS, fill_value=0)
	total = int(counts.sum())
	percentages = counts / total * 100 if total else counts.astype(float)
	return pd.DataFrame({"conteo": counts, "porcentaje": percentages}).rename_axis("sentimiento")


def validate_results(original, result):
	if len(original) != len(result):
		raise AssertionError("Cambio el numero de filas.")
	pd.testing.assert_frame_equal(original.reset_index(drop=True), result[list(original.columns)].reset_index(drop=True))
	if result.wl_fila.tolist() != list(range(len(original))):
		raise AssertionError("Cambio el orden de las filas.")
	usable = result.wl_utilizable
	predicted = result.loc[usable]
	if predicted[["wl_estrellas", "wl_sentimiento", "wl_score", "wl_tokens"]].isna().any().any():
		raise AssertionError("Faltan predicciones o metadatos en filas utilizables.")
	if not predicted.wl_estrellas.isin([1, 2, 3, 4, 5]).all():
		raise AssertionError("Estrellas fuera de rango.")
	if not predicted.wl_score.map(lambda score: math.isfinite(score) and 0 <= score <= 1).all():
		raise AssertionError("Scores invalidos.")
	if predicted.wl_estrellas.map(star_to_sentiment).tolist() != predicted.wl_sentimiento.tolist():
		raise AssertionError("Mapeo inconsistente.")
	if result.loc[~usable, ["wl_estrellas", "wl_sentimiento", "wl_score"]].notna().any().any():
		raise AssertionError("Hay predicciones en filas no utilizables.")
	if usable.any() and not math.isclose(sentiment_summary(result).porcentaje.sum(), 100, abs_tol=1e-8):
		raise AssertionError("Los porcentajes no suman 100.")


def verify_saved_csv(result, path=OUTPUT):
	saved = read_reviews(path)
	expected = result.astype(object).where(result.notna(), "").astype(str)
	if list(saved.columns) != list(expected.columns) or len(saved) != len(expected):
		raise AssertionError("Columnas o numero de filas distintos en CSV.")
	for column in expected.columns:
		if column == "wl_score":
			observed = pd.to_numeric(saved[column], errors="coerce")
			desired = pd.to_numeric(expected[column], errors="coerce")
			if not observed.isna().equals(desired.isna()):
				raise AssertionError("Scores ausentes distintos en CSV.")
			if not (observed[desired.notna()] - desired[desired.notna()]).abs().le(1e-6).all():
				raise AssertionError("Scores distintos en CSV (tolerancia 1e-6).")
		else:
			pd.testing.assert_series_equal(saved[column], expected[column], check_names=False)


def select_review_sample(result, size=20, seed=42):
	usable = result.loc[result.wl_utilizable].copy()
	random_part = usable.sample(n=min(10, len(usable), size), random_state=seed)
	random_part["motivo_seleccion"] = "aleatoria (semilla 42)"
	focused = usable.drop(random_part.index).copy()
	focused["candidato_fn"] = focused.wl_estrellas.le(2) & focused.wl_estrellas_humanas.ge(4).fillna(False)
	focused["discrepancia"] = (focused.wl_estrellas - focused.wl_estrellas_humanas).abs().fillna(0)
	focused["negativo_con_elogio"] = focused.wl_estrellas.le(2) & focused.wl_texto_limpio.str.contains(
		r"\b(?:excelente|genial|recomend\w*|amable\w*|encant\w*|perfect\w*|great|excellent|friendly|love\w*)\b",
		case=False, regex=True
	)
	focused["mixta_negacion"] = focused.wl_texto_limpio.str.contains(
		r"\b(?:no|not|never|nunca|pero|but|aunque|however|sin|neither)\b", case=False, regex=True
	)
	focused = focused.sort_values(
		["candidato_fn", "negativo_con_elogio", "discrepancia", "mixta_negacion", "wl_truncada", "wl_score", "wl_fila"],
		ascending=[False, False, False, False, False, True, True], kind="stable"
	).head(max(0, min(size, len(usable)) - len(random_part)))
	focused["motivo_seleccion"] = focused.apply(
		lambda row: "; ".join(reason for condition, reason in [
			(row.candidato_fn, "negativo frente a rating 4-5"),
			(row.negativo_con_elogio, "negativo con senal lexica positiva: leer negaciones"),
			(row.discrepancia > 0, "discrepancia de estrellas"),
			(row.mixta_negacion, "negacion u opinion mixta posible"),
			(row.wl_truncada, "texto truncado"),
			(True, "priorizacion por score relativo y fila"),
		] if condition), axis=1
	) if len(focused) else pd.Series(dtype=str)
	sample = pd.concat([random_part, focused]).drop(columns=["candidato_fn", "negativo_con_elogio", "discrepancia", "mixta_negacion"], errors="ignore")
	for column in ["interpretacion_IA", "posible_error_IA", "patron_IA", "validacion_humana", "comentario_humano"]:
		sample[column] = "pendiente"
	return sample


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default=str(INPUT))
	parser.add_argument("--output", default=str(OUTPUT))
	parser.add_argument("--text-column")
	parser.add_argument("--rating-column")
	parser.add_argument("--id-column")
	parser.add_argument("--separator", default=",")
	parser.add_argument("--encoding", default="utf-8-sig")
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--device", default="cpu")
	args = parser.parse_args(argv)
	try:
		original = read_reviews(args.input, args.separator, args.encoding)
		prepared, audit = prepare_reviews(original, args.text_column, args.rating_column, args.id_column)
		print(json.dumps(audit, ensure_ascii=False, indent=2))
		if not prepared.wl_utilizable.any():
			raise ValueError("No hay resenas utilizables: no se ejecutara inferencia.")
		output = resolve_path(args.output)
		if output.resolve() == resolve_path(args.input).resolve():
			raise ValueError("La salida no puede sobrescribir los datos originales.")
		classifier, tokenizer, limit = load_classifier(args.device)
		result = infer_reviews(prepared, classifier, tokenizer, limit, args.batch_size)
		validate_results(original, result)
		output.parent.mkdir(parents=True, exist_ok=True)
		result.to_csv(output, index=False)
		verify_saved_csv(result, output)
		print(sentiment_summary(result).to_string())
		mean = float(result.wl_estrellas.mean())
		print(f"Media predicha: {mean:.3f}; referencia externa: 4.5; diferencia: {mean - 4.5:+.3f}")
		print(f"Limite: {limit} tokens; truncadas: {int(result.wl_truncada.sum())}; CSV: {output}")
		if audit["utilizables"] != 500:
			print(f"AVISO: {audit['utilizables']} utilizables, no 500. No se eliminaron filas.")
	except (FileNotFoundError, ValueError) as error:
		parser.exit(2, f"Analisis bloqueado: {error}\n")


if __name__ == "__main__":
	main()
