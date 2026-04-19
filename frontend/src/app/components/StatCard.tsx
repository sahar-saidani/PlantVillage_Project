import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: string;
  color?: string;
}

export function StatCard({ title, value, icon: Icon, trend, color = "emerald" }: StatCardProps) {
  const colorClasses = {
    emerald: "bg-emerald-500/10 text-emerald-500",
    blue: "bg-blue-500/10 text-blue-500",
    amber: "bg-amber-500/10 text-amber-500",
    rose: "bg-rose-500/10 text-rose-500",
  };

  return (
    <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800 hover:border-gray-700 transition-all">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400 mb-1">{title}</p>
          <h3 className="text-3xl font-bold mb-2">{value}</h3>
          {trend && <p className="text-xs text-emerald-500">{trend}</p>}
        </div>
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${colorClasses[color as keyof typeof colorClasses] || colorClasses.emerald}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
}
