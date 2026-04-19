export type PredictionResult = {
  label: string;
  scores: Record<string, number> | null;
  feature_dim?: number;
};

export type LivePipelineResponse = {
  predictions: {
    classical_ml: PredictionResult | null;
    deep_learning: PredictionResult | null;
  };
  visuals: Record<string, string>;
};

export async function runLivePipeline(file: File): Promise<LivePipelineResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error ?? "Live pipeline request failed");
  }

  return (await response.json()) as LivePipelineResponse;
}
