import { useProjectData } from "../lib/use-project-data";

const featureFamilies = [
  {
    title: "Color",
    items: ["RGB channel mean and std", "RGB histograms", "HSV mean and std", "2D HSV histogram"],
  },
  {
    title: "Texture",
    items: ["GLCM contrast", "GLCM homogeneity", "GLCM energy", "GLCM correlation"],
  },
  {
    title: "Shape",
    items: ["Leaf area", "Perimeter", "Aspect ratio", "Circularity", "Hu moments summary"],
  },
  {
    title: "Contours",
    items: ["Lesion ratio", "Edge density", "Mask-based contour statistics"],
  },
];

export function FeatureExtraction() {
  const { data, loading } = useProjectData();

  if (loading || !data?.classicalSummary) {
    return <div className="text-gray-400">Loading feature extraction summary...</div>;
  }

  const featureDim = Number((data.classicalSummary as { feature_dim?: number }).feature_dim ?? 108);

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Feature Extraction</h1>
        <p className="text-gray-400">Hand-crafted descriptors used by the classical models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Feature Dimension</p>
          <h3 className="text-3xl font-bold text-emerald-500">{featureDim}</h3>
          <p className="text-xs text-gray-500 mt-2">Final vector passed to SVM and Random Forest</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Best Classical Model</p>
          <h3 className="text-2xl font-bold text-blue-500">
            {String((data.classicalSummary as { best_model_name?: string }).best_model_name ?? "svm_rbf")}
          </h3>
          <p className="text-xs text-gray-500 mt-2">Selected after validation comparison</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Pipeline Input</p>
          <h3 className="text-2xl font-bold text-rose-500">Masked RGB leaf</h3>
          <p className="text-xs text-gray-500 mt-2">Derived from preprocessing and segmentation outputs</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {featureFamilies.map((family) => (
          <div key={family.title} className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold mb-4">{family.title} Features</h2>
            <div className="space-y-2 text-sm text-gray-300">
              {family.items.map((item) => (
                <p key={item}>{item}</p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
