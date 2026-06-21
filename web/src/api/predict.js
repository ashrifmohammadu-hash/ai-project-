import { apiRequest } from "./client";

export async function predictDisease(imageFile, forceClarify = false) {
  const form = new FormData();
  form.append("image", imageFile);
  if (forceClarify) form.append("force_clarify", "true");

  return apiRequest("/predict", {
    method: "POST",
    body: form,
  });
}

export async function uploadAndPredict(imageFile, opts = {}) {
  const { forceClarify = false } = opts;

  const form = new FormData();
  form.append("image", imageFile);
  if (forceClarify) form.append("force_clarify", "true");

  const res = await apiRequest("/upload_predict", {
    method: "POST",
    body: form,
  });

  if (res && typeof res === "object" && res.prediction) {
    const result = {
      disease: res.prediction,
      display_name: (res.display_info && res.display_info.display_name) || res.display_info || res.prediction,
      plant_type: res.plant_type,
      confidence: res.confidence,
      all_predictions: res.top_predictions || [],
      treatment: res.display_info || {},
      saved_copy: res.saved_copy,
      needs_clarification: res.needs_clarification,
      clarification_questions: res.clarification_questions || [],
    };

    if (res.prediction === "unidentified") {
      const err = new Error(res.message || "The image does not contain a plant leaf. Please upload a clear photo of a leaf.");
      err.status = 422;
      throw err;
    }

    return result;
  }

  return res;
}

export async function getExplanation(imageFile) {
  const form = new FormData();
  form.append("image", imageFile);

  return apiRequest("/explain", {
    method: "POST",
    body: form,
  });
}

export async function submitAnswer(disease1, disease2, questionIndex, answer) {
  return apiRequest("/answer", {
    method: "POST",
    body: {
      disease1,
      disease2,
      question_index: questionIndex,
      answer,
    },
  });
}

export async function fetchTreatment(disease) {
  return apiRequest(`/treatment/${encodeURIComponent(disease)}`);
}

