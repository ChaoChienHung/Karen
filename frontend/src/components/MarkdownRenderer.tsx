import React, { useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const [isThoughtExpanded, setIsThoughtExpanded] = useState(true);

  if (!content) return null;

  // Extract <thought>...</thought> blocks if present
  let thoughtText = "";
  let remainingText = content;

  const thoughtStartTag = "<thought>";
  const thoughtEndTag = "</thought>";

  if (content.includes(thoughtStartTag)) {
    const startIndex = content.indexOf(thoughtStartTag);
    const endIndex = content.indexOf(thoughtEndTag);

    if (endIndex > startIndex) {
      thoughtText = content.substring(startIndex + thoughtStartTag.length, endIndex).trim();
      remainingText = (content.substring(0, startIndex) + content.substring(endIndex + thoughtEndTag.length)).trim();
    } else {
      // Thought tag is opened but not closed yet (still streaming)
      thoughtText = content.substring(startIndex + thoughtStartTag.length).trim();
      remainingText = content.substring(0, startIndex).trim();
    }
  }

  // A simple, reliable markdown-to-JSX parser that supports bullet points, bold text, headers, code, and links
  const parseMarkdownToJSX = (text: string): React.ReactNode[] => {
    const lines = text.split("\n");
    let inTable = false;
    let tableHeaders: string[] = [];
    let tableRows: string[][] = [];
    let inCodeBlock = false;
    let codeBlockLanguage = "";
    let codeBlockContent: string[] = [];

    const resultNodes: React.ReactNode[] = [];

    const parseInlineStyles = (lineText: string, keyPrefix: string): React.ReactNode => {
      // Handled simply: Bold (**text**), Italics (*text* or _text_), Code (`code`)
      let currentElement: React.ReactNode = lineText;

      // Handle Bold **
      const boldRegex = /\*\*(.*?)\*\*/g;
      const parts: React.ReactNode[] = [];
      let lastIndex = 0;
      let match;

      while ((match = boldRegex.exec(lineText)) !== null) {
        if (match.index > lastIndex) {
          parts.push(lineText.substring(lastIndex, match.index));
        }
        parts.push(<strong key={`bold-${match.index}`} className="font-bold text-slate-900 dark:text-slate-100">{match[1]}</strong>);
        lastIndex = boldRegex.lastIndex;
      }

      if (lastIndex < lineText.length) {
        parts.push(lineText.substring(lastIndex));
      }

      // Reassemble and handle inline code `
      const assembledText = parts.length > 0 ? parts : [lineText];
      const codeRegex = /`(.*?)`/g;
      const finalParts: React.ReactNode[] = [];

      assembledText.forEach((part, index) => {
        if (typeof part === "string") {
          let lastCodeIndex = 0;
          let codeMatch;
          while ((codeMatch = codeRegex.exec(part)) !== null) {
            if (codeMatch.index > lastCodeIndex) {
              finalParts.push(part.substring(lastCodeIndex, codeMatch.index));
            }
            finalParts.push(
              <code key={`inline-code-${index}-${codeMatch.index}`} className="font-mono text-xs bg-slate-200 border border-slate-300 text-indigo-700 dark:bg-slate-800 dark:border-slate-700 dark:text-indigo-300 px-1 py-0.5 rounded font-medium">
                {codeMatch[1]}
              </code>
            );
            lastCodeIndex = codeRegex.lastIndex;
          }
          if (lastCodeIndex < part.length) {
            finalParts.push(part.substring(lastCodeIndex));
          }
        } else {
          finalParts.push(part);
        }
      });

      return <span key={keyPrefix}>{finalParts.length > 0 ? finalParts : lineText}</span>;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Code Block Start/End
      if (line.trim().startsWith("```")) {
        if (inCodeBlock) {
          // End of code block
          resultNodes.push(
            <pre key={`codeblock-${i}`} className="bg-slate-900 border border-slate-700 dark:bg-slate-950/80 dark:border-slate-800 rounded-lg p-3 overflow-x-auto text-xs font-mono text-emerald-400 my-3 shadow-sm">
              <div className="flex justify-between items-center text-[10px] text-slate-400 font-sans border-b border-slate-700 dark:border-slate-800/80 pb-1.5 mb-1.5 uppercase tracking-wider">
                <span>{codeBlockLanguage || "code"}</span>
              </div>
              <code>{codeBlockContent.join("\n")}</code>
            </pre>
          );
          inCodeBlock = false;
          codeBlockContent = [];
          codeBlockLanguage = "";
        } else {
          // Start of code block
          inCodeBlock = true;
          codeBlockLanguage = line.trim().substring(3).trim();
        }
        continue;
      }

      if (inCodeBlock) {
        codeBlockContent.push(line);
        continue;
      }

      // Headers
      if (line.startsWith("# ")) {
        resultNodes.push(<h1 key={`h1-${i}`} className="text-xl font-bold font-display text-slate-900 dark:text-slate-100 mt-4 mb-2">{parseInlineStyles(line.substring(2), `h1-in-${i}`)}</h1>);
        continue;
      }
      if (line.startsWith("## ")) {
        resultNodes.push(<h2 key={`h2-${i}`} className="text-lg font-semibold font-display text-slate-900 dark:text-slate-100 mt-4 mb-2">{parseInlineStyles(line.substring(3), `h2-in-${i}`)}</h2>);
        continue;
      }
      if (line.startsWith("### ")) {
        resultNodes.push(<h3 key={`h3-${i}`} className="text-md font-semibold font-display text-slate-900 dark:text-slate-100 mt-3 mb-1">{parseInlineStyles(line.substring(4), `h3-in-${i}`)}</h3>);
        continue;
      }

      // Bullet List
      if (line.trim().startsWith("* ") || line.trim().startsWith("- ")) {
        const itemContent = line.trim().substring(2);
        resultNodes.push(
          <ul key={`ul-${i}`} className="list-disc list-inside ml-2 text-sm text-slate-800 dark:text-slate-300 mb-1">
            <li>{parseInlineStyles(itemContent, `li-${i}`)}</li>
          </ul>
        );
        continue;
      }

      // Numbered List
      const numListMatch = line.trim().match(/^(\d+)\.\s(.*)/);
      if (numListMatch) {
        const itemContent = numListMatch[2];
        resultNodes.push(
          <ol key={`ol-${i}`} className="list-decimal list-inside ml-2 text-sm text-slate-800 dark:text-slate-300 mb-1">
            <li>{parseInlineStyles(itemContent, `li-num-${i}`)}</li>
          </ol>
        );
        continue;
      }

      // Tables
      if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
        const cells = line.split("|").map(c => c.trim()).filter(c => c !== "");
        if (line.includes("---") || line.includes("-|-")) {
          // Table separator row, ignore
          continue;
        }
        if (!inTable) {
          inTable = true;
          tableHeaders = cells;
          tableRows = [];
        } else {
          tableRows.push(cells);
        }
        continue;
      } else if (inTable) {
        // Table finished, render table node
        resultNodes.push(
          <div key={`table-${i}`} className="overflow-x-auto my-3 border border-slate-300 dark:border-slate-800 rounded-lg shadow-sm">
            <table className="min-w-full divide-y divide-slate-300 dark:divide-slate-800 text-xs">
              <thead className="bg-slate-100 dark:bg-slate-900/50">
                <tr>
                  {tableHeaders.map((hdr, hIdx) => (
                    <th key={`th-${hIdx}`} className="px-3 py-2 text-left font-semibold text-slate-900 dark:text-slate-300 tracking-wider">
                      {hdr}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-950/20">
                {tableRows.map((row, rIdx) => (
                  <tr key={`tr-${rIdx}`} className="hover:bg-slate-50 dark:hover:bg-slate-900/20">
                    {row.map((cell, cIdx) => (
                      <td key={`td-${cIdx}`} className="px-3 py-2 text-slate-800 dark:text-slate-400 whitespace-normal">
                        {parseInlineStyles(cell, `cell-${rIdx}-${cIdx}`)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        inTable = false;
        tableHeaders = [];
        tableRows = [];
      }

      // Paragraph / Blank space
      if (line.trim() === "") {
        resultNodes.push(<div key={`empty-${i}`} className="h-2" />);
      } else {
        resultNodes.push(
          <p key={`p-${i}`} className="text-sm text-slate-800 dark:text-slate-300 leading-relaxed mb-1.5">
            {parseInlineStyles(line, `p-in-${i}`)}
          </p>
        );
      }
    }

    // Edge case if file ends while in a table
    if (inTable && tableHeaders.length > 0) {
      resultNodes.push(
        <div key="table-end" className="overflow-x-auto my-3 border border-slate-300 dark:border-slate-800 rounded-lg shadow-sm">
          <table className="min-w-full divide-y divide-slate-300 dark:divide-slate-800 text-xs">
            <thead className="bg-slate-100 dark:bg-slate-900/50">
              <tr>
                {tableHeaders.map((hdr, hIdx) => (
                  <th key={`th-end-${hIdx}`} className="px-3 py-2 text-left font-semibold text-slate-900 dark:text-slate-300 tracking-wider">
                    {hdr}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800 bg-white dark:bg-slate-950/20">
              {tableRows.map((row, rIdx) => (
                <tr key={`tr-end-${rIdx}`} className="hover:bg-slate-50 dark:hover:bg-slate-900/20">
                  {row.map((cell, cIdx) => (
                    <td key={`td-end-${cIdx}`} className="px-3 py-2 text-slate-800 dark:text-slate-400">
                      {parseInlineStyles(cell, `cell-end-${rIdx}-${cIdx}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    return resultNodes;
  };

  return (
    <div className="space-y-3 select-text selection:bg-indigo-500">
      {/* Collapsible reasoning thoughts block */}
      {thoughtText && (
        <div id="thought-block" className="border border-slate-300 dark:border-slate-800/80 rounded-xl bg-white dark:bg-slate-950/40 shadow-sm overflow-hidden transition-all duration-300">
          <button
            onClick={() => setIsThoughtExpanded(!isThoughtExpanded)}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-100 dark:bg-slate-900/60 text-indigo-700 dark:text-indigo-300 font-mono text-xs hover:bg-slate-200 dark:hover:bg-slate-900/80 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 animate-pulse text-indigo-600 dark:text-indigo-400" />
              <span className="font-bold tracking-wide">Thinking Process</span>
              <span className="text-[10px] text-slate-600 dark:text-slate-500 bg-slate-200 dark:bg-slate-800/50 px-1.5 py-0.5 rounded font-sans">
                {thoughtText.length} characters
              </span>
            </div>
            {isThoughtExpanded ? (
              <ChevronUp className="w-3.5 h-3.5 text-slate-600 dark:text-slate-400" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 text-slate-600 dark:text-slate-400" />
            )}
          </button>
          
          {isThoughtExpanded && (
            <div className="p-4 border-t border-slate-200 dark:border-slate-900/80 text-xs font-mono text-slate-700 dark:text-slate-400 bg-slate-50 dark:bg-slate-950/60 overflow-y-auto max-h-48 leading-relaxed whitespace-pre-wrap selection:bg-indigo-900/50 select-text scrollbar-thin">
              {thoughtText}
            </div>
          )}
        </div>
      )}

      {/* Actual Markdown formatted answer */}
      <div id="answer-content" className="markdown-body text-slate-800 dark:text-slate-300 font-sans tracking-normal leading-relaxed">
        {parseMarkdownToJSX(remainingText)}
      </div>
    </div>
  );
}
