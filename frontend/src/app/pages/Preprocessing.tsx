import { CompactImageImport } from "../components/CompactImageImport";
import { useProjectData } from "../lib/use-project-data";
import { useLivePipeline } from "../lib/use-live-pipeline";

export function Preprocessing() {
  const { data, loading } = useProjectData();
  const live = useLivePipeline();

  if (loading || !data) {
    return <div className="text-gray-400">Loading preprocessing outputs...</div>;
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Preprocessing</h1>
        <p className="text-gray-400">Run live preprocessing on an imported image, then compare with exported examples.</p>
      </div>

      <CompactImageImport loading={live.loading} onFileSelect={live.run} subtitle="Shows original, preprocessing, grayscale, and advanced preprocessing outputs." />

      {live.error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {live.error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Imported Preview</h2>
          {live.previewUrl ? (
            <img src={live.previewUrl} alt="Imported preview" className="w-full rounded-lg border border-gray-800" />
          ) : (
            <div className="flex min-h-64 items-center justify-center rounded-lg border border-gray-800 bg-gray-950 text-sm text-gray-500">
              No image selected yet
            </div>
          )}
        </div>

        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Live Preprocessed Output</h2>
          {live.result?.visuals.preprocessed ? (
            <img
              src={`data:image/png;base64,${live.result.visuals.preprocessed}`}
              alt="Live preprocessed output"
              className="w-full rounded-lg border border-gray-800"
            />
          ) : (
            <div className="flex min-h-64 items-center justify-center rounded-lg border border-gray-800 bg-gray-950 text-sm text-gray-500">
              Upload an image to run preprocessing
            </div>
          )}
        </div>
      </div>

      {live.result && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { label: "Original", key: "original" },
            { label: "Grayscale", key: "grayscale" },
            { label: "Advanced preprocessing", key: "advanced_preprocessed" },
          ].map((item) => (
            <div key={item.key} className="bg-[#1a1d27] rounded-xl p-4 border border-gray-800">
              <h3 className="text-sm font-semibold mb-3 text-gray-300">{item.label}</h3>
              <img
                src={`data:image/png;base64,${live.result.visuals[item.key]}`}
                alt={item.label}
                className="w-full rounded-lg border border-gray-800"
              />
            </div>
          ))}
        </div>
      )}

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Applied Steps</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-300">
          <div className="rounded-lg bg-gray-900/50 p-4">RGB input resized to 224x224</div>
          <div className="rounded-lg bg-gray-900/50 p-4">Classical preprocessing with denoising and CLAHE</div>
          <div className="rounded-lg bg-gray-900/50 p-4">Advanced branch with bilateral filter and sharpening</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {data.labeledExamples.map((item) => (
          <div key={item.src} className="bg-[#1a1d27] rounded-xl p-4 border border-gray-800">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">{item.label}</h3>
            <img
              src={item.src}
              alt={item.label}
              className="w-full rounded-lg border border-gray-800 bg-gray-950"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
