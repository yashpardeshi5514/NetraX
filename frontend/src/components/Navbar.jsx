import {
  Brain,
  LayoutDashboard,
  History,
  LogOut,
  ScanLine,
} from "lucide-react";

export default function Navbar({
  user,
  currentPage,
  onDashboard,
  onAnalysis,
  onHistory,
  onLogout,
}) {
  const navItems = [
    {
      key: "dashboard",
      label: "Dashboard",
      icon: LayoutDashboard,
      action: onDashboard,
    },
    {
      key: "analysis",
      label: "Analysis",
      icon: ScanLine,
      action: onAnalysis,
    },
    {
      key: "history",
      label: "History",
      icon: History,
      action: onHistory,
    },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-[#D8D2C8] bg-[#F7F2EB]/95 backdrop-blur">
      <div className="max-w-7xl mx-auto px-5 md:px-8">
        <div className="min-h-[76px] flex items-center justify-between gap-4">
          <button
            onClick={onDashboard}
            className="flex items-center gap-3 shrink-0"
          >
            <div className="h-11 w-11 rounded-xl bg-[#8B9A6E] text-[#F7F2EB] flex items-center justify-center shadow-sm">
              <Brain size={23} />
            </div>

            <div className="hidden sm:block text-left">
              <p className="font-bold text-lg leading-tight text-[#2F3728]">
                NetraX
              </p>
              <p className="text-xs text-[#69705F]">
                Retinal AI Decision Support
              </p>
            </div>
          </button>

          <nav className="flex items-center gap-1">
            {navItems.map(({ key, label, icon: Icon, action }) => (
              <button
                key={key}
                onClick={action}
                className={`inline-flex items-center gap-2 rounded-xl px-3 md:px-4 py-2.5 text-sm font-semibold transition ${
                  currentPage === key
                    ? "bg-[#8B9A6E] text-[#F7F2EB] shadow-sm"
                    : "text-[#69705F] hover:bg-[#EAE2D6] hover:text-[#2F3728]"
                }`}
              >
                <Icon size={16} />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}

            <div className="hidden md:block h-8 w-px bg-[#D8D2C8] mx-2" />

            <div className="hidden lg:block text-right mr-2 max-w-[190px]">
              <p className="text-sm font-medium text-[#2F3728] truncate">
                {user?.email}
              </p>
              <p className="text-[11px] text-[#69705F]">
                Authenticated
              </p>
            </div>

            <button
              onClick={onLogout}
              title="Logout"
              className="inline-flex items-center gap-2 rounded-xl border border-[#D8D2C8] bg-[#EEEEEE] px-3 py-2.5 text-sm font-semibold text-[#4D5545] hover:bg-[#EAE2D6] hover:border-[#8B9A6E] hover:text-[#2F3728] transition"
            >
              <LogOut size={16} />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}
