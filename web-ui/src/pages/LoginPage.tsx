import React, { useState } from "react";
import { KeyRound, Lock, Sparkles } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { Button, Card, Input } from "@/components/ui/primitives";

export function LoginPage() {
  const { refreshMe } = useAuth();
  const [mode, setMode] = useState<"password" | "apikey" | "register">("password");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setInfo("");
    setSubmitting(true);
    try {
      if (mode === "register") {
        await api.register({ username: username.trim(), password });
        setInfo("注册成功，请联系管理员审批后登录");
        setMode("password");
      } else if (mode === "apikey") {
        const res = await api.login({ api_key: apiKey.trim() });
        if (!res.success) {
          setError(res.message || "API Key 无效");
        } else {
          await refreshMe();
        }
      } else {
        const res = await api.login({ username: username.trim(), password });
        if (!res.success) {
          setError(res.message || "用户名或密码错误");
        } else {
          await refreshMe();
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 px-4 dark:bg-zinc-950">
      <Card className="w-full max-w-sm p-6 shadow-sm">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <h1 className="text-base font-semibold">TeamMemory</h1>
            <p className="text-xs text-zinc-500">AI 记忆系统控制台 · 多租户隔离</p>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 rounded-lg bg-zinc-100 p-1 text-xs font-medium dark:bg-zinc-900">
          <button
            type="button"
            onClick={() => {
              setMode("password");
              setError("");
            }}
            className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 transition-colors cursor-pointer ${
              mode !== "apikey"
                ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                : "text-zinc-500"
            }`}
          >
            <Lock className="h-3.5 w-3.5" />
            账号密码
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("apikey");
              setError("");
            }}
            className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 transition-colors cursor-pointer ${
              mode === "apikey"
                ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                : "text-zinc-500"
            }`}
          >
            <KeyRound className="h-3.5 w-3.5" />
            API Key
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          {mode === "apikey" ? (
            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-300">
                租户 API Key
              </label>
              <Input
                type="password"
                autoFocus
                placeholder="粘贴你的 API Key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
            </div>
          ) : (
            <>
              <div>
                <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  用户名
                </label>
                <Input
                  autoFocus
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  密码
                </label>
                <Input
                  type="password"
                  placeholder="••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </>
          )}

          {error && <p className="text-xs text-red-500">{error}</p>}
          {info && <p className="text-xs text-emerald-600 dark:text-emerald-400">{info}</p>}

          <Button type="submit" className="w-full" disabled={submitting}>
            {submitting
              ? "处理中..."
              : mode === "register"
                ? "提交注册"
                : "登录控制台"}
          </Button>
        </form>

        {mode !== "apikey" && (
          <div className="mt-4 text-center text-xs text-zinc-500">
            {mode === "register" ? (
              <button
                type="button"
                className="underline hover:text-zinc-800 dark:hover:text-zinc-200 cursor-pointer"
                onClick={() => setMode("password")}
              >
                已有账号？返回登录
              </button>
            ) : (
              <button
                type="button"
                className="underline hover:text-zinc-800 dark:hover:text-zinc-200 cursor-pointer"
                onClick={() => setMode("register")}
              >
                新成员？自助注册
              </button>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
