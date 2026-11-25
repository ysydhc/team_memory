import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { AppShell, type NavTab } from "@/components/AppShell";
import { CommandPalette } from "@/components/CommandPalette";
import { MemoryDrawer, type DrawerTarget } from "@/components/MemoryDrawer";
import { LoginPage } from "@/pages/LoginPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { MemoriesPage } from "@/pages/MemoriesPage";
import { AccessPage } from "@/pages/AccessPage";
import { OpsPage } from "@/pages/OpsPage";

function readInitialTab(): NavTab {
  const path = window.location.pathname.replace(/^\/+/, "");
  if (path === "overview" || path === "memories" || path === "access" || path === "ops") {
    return path;
  }
  return "memories";
}

export function App() {
  const { user, loading, isAdmin, currentTenant } = useAuth();
  const queryClient = useQueryClient();

  const [activeTab, setActiveTab] = useState<NavTab>(readInitialTab);
  const [memoriesInitialView, setMemoriesInitialView] = useState<
    "experiences" | "drafts" | "archives" | "personal"
  >("experiences");
  const [cmdOpen, setCmdOpen] = useState(false);
  const [drawerTarget, setDrawerTarget] = useState<DrawerTarget>(null);

  // Global Cmd+K / Ctrl+K shortcut
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCmdOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Sync URL path on navigation and browser back/forward (popstate)
  const handleNavigate = (tab: NavTab) => {
    setActiveTab(tab);
    if (window.location.pathname !== `/${tab}`) {
      window.history.pushState({}, "", `/${tab}`);
    }
  };

  useEffect(() => {
    const onPopState = () => {
      setActiveTab(readInitialTab());
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  // Redirect non-admin away from Ops if tenant role changes
  useEffect(() => {
    if (!isAdmin && activeTab === "ops") {
      setActiveTab("memories");
    }
  }, [isAdmin, activeTab]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-zinc-50 text-xs text-zinc-400 dark:bg-zinc-950">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        正在加载 TeamMemory...
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  return (
    <>
      <AppShell
        activeTab={activeTab}
        onNavigate={handleNavigate}
        onOpenCommandPalette={() => setCmdOpen(true)}
      >
        {activeTab === "overview" && (
          <OverviewPage
            onGoToDrafts={() => {
              setMemoriesInitialView("drafts");
              handleNavigate("memories");
            }}
          />
        )}
        {activeTab === "memories" && (
          <MemoriesPage
            key={`${currentTenant?.id || "default"}-${memoriesInitialView}`}
            initialView={memoriesInitialView}
            onOpenDrawer={(target) => setDrawerTarget(target)}
          />
        )}
        {activeTab === "access" && <AccessPage />}
        {activeTab === "ops" && isAdmin && <OpsPage />}
      </AppShell>

      <CommandPalette
        open={cmdOpen}
        onClose={() => setCmdOpen(false)}
        onSelectExperience={(exp) => setDrawerTarget({ kind: "experience", item: exp })}
      />

      <MemoryDrawer
        target={drawerTarget}
        onClose={() => setDrawerTarget(null)}
        onSwitchTarget={(t) => setDrawerTarget(t)}
        onMutated={() => {
          void queryClient.invalidateQueries();
        }}
      />
    </>
  );
}

export default App;
