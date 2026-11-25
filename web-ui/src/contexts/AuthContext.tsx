import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  getStoredTenantId,
  setStoredTenantId,
  type AuthMeResponse,
  type TenantSummary,
} from "@/lib/api";

interface AuthContextValue {
  user: AuthMeResponse | null;
  loading: boolean;
  isAdmin: boolean;
  canWrite: boolean;
  currentTenant: TenantSummary | null;
  tenants: TenantSummary[];
  refreshMe: () => Promise<void>;
  switchTenant: (tenantId: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  isAdmin: false,
  canWrite: false,
  currentTenant: null,
  tenants: [],
  refreshMe: async () => {},
  switchTenant: async () => {},
  logout: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<AuthMeResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshMe = useCallback(async () => {
    try {
      const me = await api.me();
      setUser(me);
      if (me.tenant_id && getStoredTenantId() !== me.tenant_id) {
        setStoredTenantId(me.tenant_id);
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 403 && getStoredTenantId()) {
        // Stored tenant may no longer be accessible; clear and retry once
        setStoredTenantId(null);
        try {
          const me = await api.me();
          setUser(me);
          setStoredTenantId(me.tenant_id);
          return;
        } catch {
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshMe();
  }, [refreshMe]);

  const switchTenant = useCallback(
    async (tenantId: string) => {
      setStoredTenantId(tenantId);
      await refreshMe();
      // Invalidate all queries so every page reloads strictly for the new tenant
      await queryClient.invalidateQueries();
    },
    [queryClient, refreshMe],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // ignore
    }
    setStoredTenantId(null);
    setUser(null);
    queryClient.clear();
  }, [queryClient]);

  const role = user?.role || "viewer";
  const isAdmin = role === "admin";
  const canWrite = role === "admin" || role === "member" || role === "editor";
  const tenants = user?.tenants || [];
  const currentTenant =
    user?.current_tenant || tenants.find((t) => t.id === user?.tenant_id) || null;

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAdmin,
        canWrite,
        currentTenant,
        tenants,
        refreshMe,
        switchTenant,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
