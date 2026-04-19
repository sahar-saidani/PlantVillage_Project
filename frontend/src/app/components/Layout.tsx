import { Outlet, NavLink } from "react-router";
import { LayoutDashboard, ImageIcon, Scissors, Database, BrainCircuit, Network, Leaf } from "lucide-react";

const navItems = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { path: "/preprocessing", label: "Preprocessing", icon: ImageIcon },
  { path: "/segmentation", label: "Segmentation", icon: Scissors },
  { path: "/feature-extraction", label: "Feature Extraction", icon: Database },
  { path: "/classification", label: "Classification (ML)", icon: BrainCircuit },
  { path: "/deep-learning", label: "Deep Learning (DL)", icon: Network },
];

export function Layout() {
  return (
    <div className="flex h-screen bg-[#0f1117] text-white">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1a1d27] border-r border-gray-800 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500/10 rounded-lg flex items-center justify-center">
              <Leaf className="w-6 h-6 text-emerald-500" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">PlantVision AI</h1>
              <p className="text-xs text-gray-400">Disease Detection</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.exact}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? "bg-emerald-500/10 text-emerald-500"
                    : "text-gray-400 hover:bg-gray-800/50 hover:text-white"
                }`
              }
            >
              <item.icon className="w-5 h-5" />
              <span className="text-sm font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="px-4 py-3 bg-gray-800/50 rounded-lg">
            <p className="text-xs text-gray-400">System Status</p>
            <div className="flex items-center gap-2 mt-1">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-emerald-500">All Systems Online</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-[#1a1d27] border-b border-gray-800 flex items-center px-8">
          <div className="flex items-center gap-3">
            <Leaf className="w-5 h-5 text-emerald-500" />
            <h2 className="text-lg font-semibold">PlantVision AI</h2>
          </div>
          <div className="ml-auto flex items-center gap-4">
            <div className="text-sm text-gray-400">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
