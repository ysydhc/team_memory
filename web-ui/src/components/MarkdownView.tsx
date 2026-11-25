import React from "react";
import { cn } from "@/lib/utils";

/**
 * Lightweight, XSS-safe Markdown renderer (zero dangerouslySetInnerHTML).
 * Supports headings, fenced code blocks, inline code, bold, lists, blockquotes, and links.
 */

function renderInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const tokenRe = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  let lastIdx = 0;
  let match: RegExpExecArray | null;
  let k = 0;

  while ((match = tokenRe.exec(text)) !== null) {
    if (match.index > lastIdx) {
      nodes.push(text.slice(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <code
          key={k++}
          className="rounded bg-zinc-200/70 px-1 py-0.5 font-mono text-[12px] text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(
        <strong key={k++} className="font-semibold text-zinc-900 dark:text-zinc-100">
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith("[")) {
      const m = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
      if (m) {
        nodes.push(
          <a
            key={k++}
            href={m[2]}
            target="_blank"
            rel="noreferrer"
            className="text-sky-600 underline decoration-sky-400/50 hover:text-sky-700 dark:text-sky-400"
          >
            {m[1]}
          </a>,
        );
      } else {
        nodes.push(token);
      }
    }
    lastIdx = match.index + token.length;
  }
  if (lastIdx < text.length) {
    nodes.push(text.slice(lastIdx));
  }
  return nodes;
}

export function MarkdownView({ content, className }: { content: string; className?: string }) {
  const lines = (content || "").replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.trimStart().startsWith("```")) {
      const lang = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++;
      blocks.push(
        <div
          key={key++}
          className="my-2 overflow-hidden rounded-lg border border-zinc-200/80 bg-zinc-950 dark:border-zinc-800"
        >
          {lang && (
            <div className="border-b border-zinc-800 px-3 py-1 font-mono text-[10px] text-zinc-400">
              {lang}
            </div>
          )}
          <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed text-zinc-100">
            {codeLines.join("\n")}
          </pre>
        </div>,
      );
      continue;
    }

    // Blank line
    if (!line.trim()) {
      i++;
      continue;
    }

    // Headings
    const hMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const cls =
        level === 1
          ? "mt-3 mb-1.5 text-base font-semibold"
          : level === 2
            ? "mt-3 mb-1 text-sm font-semibold"
            : "mt-2 mb-1 text-xs font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300";
      blocks.push(
        <div key={key++} className={cls}>
          {renderInline(hMatch[2])}
        </div>,
      );
      i++;
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) {
      blocks.push(<hr key={key++} className="my-3 border-zinc-200 dark:border-zinc-800" />);
      i++;
      continue;
    }

    // Blockquote
    if (line.trimStart().startsWith("> ")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
        quoteLines.push(lines[i].trimStart().slice(2));
        i++;
      }
      blocks.push(
        <blockquote
          key={key++}
          className="my-2 border-l-2 border-zinc-300 pl-3 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-400"
        >
          {renderInline(quoteLines.join(" "))}
        </blockquote>,
      );
      continue;
    }

    // Unordered or ordered list
    if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
      const isOrdered = /^\s*\d+\.\s+/.test(line);
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, ""));
        i++;
      }
      const ListTag = isOrdered ? "ol" : "ul";
      blocks.push(
        <ListTag
          key={key++}
          className={cn(
            "my-1.5 space-y-1 pl-5 text-sm leading-relaxed",
            isOrdered ? "list-decimal" : "list-disc",
          )}
        >
          {items.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ListTag>,
      );
      continue;
    }

    // Paragraph
    const paraLines: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trimStart().startsWith("```") &&
      !/^#{1,4}\s+/.test(lines[i]) &&
      !/^\s*([-*+]|\d+\.)\s+/.test(lines[i]) &&
      !lines[i].trimStart().startsWith("> ")
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="my-1.5 text-sm leading-relaxed whitespace-pre-wrap">
        {renderInline(paraLines.join("\n"))}
      </p>,
    );
  }

  return <div className={cn("text-zinc-800 dark:text-zinc-200", className)}>{blocks}</div>;
}
