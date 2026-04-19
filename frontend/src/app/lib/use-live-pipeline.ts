import { useState } from "react";
import { LivePipelineResponse, runLivePipeline } from "./live-pipeline";

export function useLivePipeline() {
  const [result, setResult] = useState<LivePipelineResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (file: File) => {
    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    try {
      const payload = await runLivePipeline(file);
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return { result, previewUrl, loading, error, run };
}
