import { useProjectData } from "../lib/use-project-data";

export function Segmentation() {
  const { data, loading } = useProjectData();

  if (loading || !data?.segmentationBenchmark) {
    return <div className="text-gray-400">Loading segmentation benchmark...</div>;
  }

  const methods = Object.entries(data.segmentationBenchmark.methods);

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Segmentation</h1>
        <p className="text-gray-400">
          Benchmark of classical and advanced segmentation methods, with exported example images from the pipeline.
        </p>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Best Method</h2>
        <p className="text-2xl font-bold text-emerald-500">
          {data.segmentationBenchmark.best_method?.replaceAll("_", " ") ?? "n/a"}
        </p>
        <p className="text-sm text-gray-400 mt-2">
          Evaluated on {data.segmentationBenchmark.num_images_evaluated} training images with an
          unsupervised quality score.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {data.labeledExamples.map((item) => (
          <div key={item.src} className="bg-[#1a1d27] rounded-xl p-4 border border-gray-800">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">{item.label}</h3>
            <img src={item.src} alt={item.label} className="w-full rounded-lg border border-gray-800 bg-gray-950" />
            <p className="mt-3 text-xs text-gray-400">
              Each exported strip contains original image, preprocessing, masks, segmented outputs, edges, and contours.
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {methods.map(([name, metrics]) => (
          <div key={name} className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
            <div className="flex items-start justify-between gap-4 mb-3">
              <div>
                <h3 className="text-lg font-semibold">{name.replaceAll("_", " ")}</h3>
                <p className="text-sm text-gray-400 mt-1">{metrics.description}</p>
              </div>
              <span className="rounded-full bg-emerald-500/10 px-3 py-1 text-sm font-semibold text-emerald-500">
                {(metrics.quality_score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Green contrast</span>
                <span>{(metrics.green_contrast * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Largest component ratio</span>
                <span>{(metrics.largest_component_ratio * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Edge alignment</span>
                <span>{(metrics.edge_alignment * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Area ratio</span>
                <span>{(metrics.area_ratio * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Notes</h2>
        <div className="space-y-2 text-sm text-gray-300">
          {data.segmentationBenchmark.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      </div>
    </div>
  );
}
