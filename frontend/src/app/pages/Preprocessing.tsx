import { useState } from "react";
import { Upload } from "lucide-react";
import { useProjectData } from "../lib/use-project-data";

export function Preprocessing() {
  const { data, loading } = useProjectData();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setPreviewUrl(URL.createObjectURL(file));
  };

  if (loading || !data) {
    return <div className="text-gray-400">Loading preprocessing outputs...</div>;
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Preprocessing</h1>
        <p className="text-gray-400">
          Upload preview on the left, and real preprocessing outputs exported from the Python pipeline below.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <label className="bg-[#1a1d27] rounded-xl p-6 border border-dashed border-gray-700 hover:border-emerald-500/40 transition-all cursor-pointer block">
          <div className="flex flex-col items-center justify-center text-center min-h-64">
            <Upload className="w-10 h-10 text-emerald-500 mb-3" />
            <p className="font-semibold">Import image</p>
            <p className="text-sm text-gray-400 mt-2">
              This preview is local to the frontend. Processed step images below come from exported pipeline artifacts.
            </p>
            <input type="file" accept="image/*" className="hidden" onChange={onFileChange} />
          </div>
        </label>

        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Imported Preview</h2>
          {previewUrl ? (
            <img src={previewUrl} alt="Imported preview" className="w-full rounded-lg border border-gray-800" />
          ) : (
            <div className="flex min-h-64 items-center justify-center rounded-lg border border-gray-800 bg-gray-950 text-sm text-gray-500">
              No image selected yet
            </div>
          )}
        </div>
      </div>

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
