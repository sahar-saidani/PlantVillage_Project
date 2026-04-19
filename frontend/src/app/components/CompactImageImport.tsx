import type { ChangeEvent } from "react";
import { Upload } from "lucide-react";

type CompactImageImportProps = {
  loading?: boolean;
  onFileSelect: (file: File) => void | Promise<void>;
  subtitle?: string;
};

export function CompactImageImport({
  loading = false,
  onFileSelect,
  subtitle = "Upload a tomato leaf image to run the live Python pipeline.",
}: CompactImageImportProps) {
  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    void onFileSelect(file);
    event.target.value = "";
  };

  return (
    <label className="flex items-center gap-4 rounded-xl border border-dashed border-gray-700 bg-[#1a1d27] px-4 py-4 transition-all hover:border-emerald-500/40 cursor-pointer">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
        <Upload className="h-5 w-5 text-emerald-500" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="font-semibold text-white">{loading ? "Processing image..." : "Import image"}</p>
        <p className="truncate text-sm text-gray-400">{subtitle}</p>
      </div>
      <div className="rounded-lg bg-gray-900/70 px-3 py-2 text-sm text-gray-300">
        {loading ? "Running..." : "Choose file"}
      </div>
      <input type="file" accept="image/*" className="hidden" onChange={onChange} />
    </label>
  );
}
