/**
 * Built-in coding tool definitions in OpenAI function calling format.
 * Injected into request.tools when builtinToolsEnabled = true.
 * Server merges these with MCP tools automatically.
 */

export interface ToolDefinition {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: {
      type: 'object'
      properties: Record<string, any>
      required: string[]
    }
  }
}

export const BUILTIN_TOOLS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file with line numbers. Returns up to 2000 lines by default. Use offset/limit for large files.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path relative to working directory'
          },
          offset: {
            type: 'integer',
            description: 'Start reading from this line number (1-based). Default: 1'
          },
          limit: {
            type: 'integer',
            description: 'Maximum number of lines to return. Default: 2000'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a file. Creates the file and parent directories if they don\'t exist, overwrites if they do.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path relative to working directory'
          },
          content: {
            type: 'string',
            description: 'Content to write to the file'
          }
        },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'edit_file',
      description: 'Edit a file by finding and replacing text. The search_text must match exactly (including indentation). Use read_file first to see the current content. Set replace_all=true to replace ALL occurrences (useful for renaming variables/functions).',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path relative to working directory'
          },
          search_text: {
            type: 'string',
            description: 'Exact text to find in the file (must be unique unless replace_all=true)'
          },
          replacement_text: {
            type: 'string',
            description: 'Text to replace the search_text with'
          },
          replace_all: {
            type: 'boolean',
            description: 'Replace ALL occurrences instead of just the first. Default: false'
          }
        },
        required: ['path', 'search_text', 'replacement_text']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'list_directory',
      description: 'List files and directories at a path. Shows file types and sizes.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Directory path relative to working directory. Use "." for the root.'
          },
          recursive: {
            type: 'boolean',
            description: 'List recursively (default: false). Use sparingly on large trees.'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_files',
      description: 'Search file contents for a pattern using ripgrep. Returns matching lines with file paths and line numbers.',
      parameters: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'Search pattern (supports regex)'
          },
          path: {
            type: 'string',
            description: 'Directory to search in, relative to working directory. Default: "."'
          },
          glob: {
            type: 'string',
            description: 'File glob filter (e.g., "*.ts", "src/**/*.py")'
          }
        },
        required: ['pattern']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'find_files',
      description: 'Find files by name pattern. Returns matching file paths. Useful for finding files when you don\'t know their exact location.',
      parameters: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'File name pattern (glob syntax, e.g., "*.test.ts", "package.json", "**/*.py")'
          },
          path: {
            type: 'string',
            description: 'Directory to search in, relative to working directory. Default: "."'
          }
        },
        required: ['pattern']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'file_info',
      description: 'Get metadata about a file or directory: size, type (file/directory/symlink), last modified time, permissions.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Path relative to working directory'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'create_directory',
      description: 'Create a directory and any necessary parent directories.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Directory path relative to working directory'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'delete_file',
      description: 'Delete a file or empty directory. Use with caution — this cannot be undone.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Path relative to working directory'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'move_file',
      description: 'Move or rename a file or directory.',
      parameters: {
        type: 'object',
        properties: {
          source: {
            type: 'string',
            description: 'Source path relative to working directory'
          },
          destination: {
            type: 'string',
            description: 'Destination path relative to working directory'
          }
        },
        required: ['source', 'destination']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'copy_file',
      description: 'Copy a file to a new location.',
      parameters: {
        type: 'object',
        properties: {
          source: {
            type: 'string',
            description: 'Source file path relative to working directory'
          },
          destination: {
            type: 'string',
            description: 'Destination file path relative to working directory'
          }
        },
        required: ['source', 'destination']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_command',
      description: 'Execute a shell command in the working directory. Has a 60 second timeout. Returns stdout, stderr, and exit code.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'Shell command to execute (runs via /bin/sh)'
          }
        },
        required: ['command']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Search the web using Brave Search. Returns titles, URLs, and descriptions of matching pages.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query'
          },
          count: {
            type: 'integer',
            description: 'Number of results (1-20, default: 5)'
          }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'fetch_url',
      description: 'Fetch a URL and return its content as text. HTML is automatically converted to readable text. Useful for reading documentation, API responses, or web pages.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'Full URL to fetch (https://...)'
          },
          max_length: {
            type: 'integer',
            description: 'Maximum characters to return (default: 20000)'
          }
        },
        required: ['url']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ddg_search',
      description: 'Search the web using DuckDuckGo (free, no API key required). Returns titles, URLs, and snippets.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query'
          },
          count: {
            type: 'integer',
            description: 'Number of results (1-10, default: 5)'
          }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'patch_file',
      description: 'Apply a unified diff patch to a file. Useful for making multiple related edits at once. The patch should be in standard unified diff format (--- a/file, +++ b/file, @@ hunks).',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path relative to working directory'
          },
          patch: {
            type: 'string',
            description: 'Unified diff patch content (with @@ hunk headers, - for removed lines, + for added lines)'
          }
        },
        required: ['path', 'patch']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'batch_edit',
      description: 'Apply multiple find-and-replace edits to a file in a single call. More efficient than multiple edit_file calls. Each edit is applied sequentially.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path relative to working directory'
          },
          edits: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                search_text: { type: 'string', description: 'Exact text to find' },
                replacement_text: { type: 'string', description: 'Text to replace with' }
              },
              required: ['search_text', 'replacement_text']
            },
            description: 'Array of {search_text, replacement_text} edits to apply in order'
          }
        },
        required: ['path', 'edits']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'get_diagnostics',
      description: 'Run diagnostics on a file or project: type checking (TypeScript), linting, or syntax validation. Returns errors and warnings with file locations.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File or directory path relative to working directory. Use "." for project root.'
          },
          tool: {
            type: 'string',
            description: 'Diagnostic tool to run: "auto" (detect from project), "tsc" (TypeScript), "eslint", "python" (py_compile). Default: "auto"'
          }
        },
        required: ['path']
      }
    }
  }
]

const BUILTIN_TOOL_NAMES = new Set(BUILTIN_TOOLS.map(t => t.function.name))

/** Check if a tool name is a built-in tool */
export function isBuiltinTool(toolName: string): boolean {
  return BUILTIN_TOOL_NAMES.has(toolName)
}

/** Agentic system prompt injected when built-in tools are enabled and no custom system prompt is set */
export const AGENTIC_SYSTEM_PROMPT = `You are an expert software engineer with direct access to the project's file system, terminal, and the web.

Use tool calls to complete tasks. Chain multiple tool calls as needed — don't stop after a single tool call if more work is required.

CRITICAL RULE: After using tools to gather information (reading files, searching, browsing the web, running commands), you MUST ALWAYS provide a substantive response to the user explaining what you found, what you did, or answering their question. NEVER stop after just executing tools — the user needs your analysis, summary, or explanation of the results.

TOOLS:
File I/O:
  read_file(path, offset?, limit?) — read file with line numbers (paginated, 2000 lines max per call)
  write_file(path, content) — create or overwrite a file
  edit_file(path, search_text, replacement_text, replace_all?) — find-and-replace (ALWAYS read_file first). Set replace_all=true for renaming.
  patch_file(path, patch) — apply a unified diff patch for complex multi-hunk edits
  batch_edit(path, edits[]) — multiple find-and-replace edits in one call
  copy_file(source, destination) — copy a file
  move_file(source, destination) — move/rename a file or directory
  delete_file(path) — delete a file or empty directory
  create_directory(path) — create directories recursively
  file_info(path) — get size, type, modified time, permissions
Search & Navigate:
  list_directory(path, recursive?) — list files/dirs with sizes
  search_files(pattern, path?, glob?) — search file contents (regex, ripgrep)
  find_files(pattern, path?) — find files by name (glob)
  get_diagnostics(path?, tool?) — run type checking / linting (auto-detects tsc, eslint, python)
Terminal:
  run_command(command) — execute a shell command (60s timeout)
Web:
  ddg_search(query, count?) — search the web (DuckDuckGo, free, no key needed)
  web_search(query, count?) — search the web (Brave Search, requires API key)
  fetch_url(url, max_length?) — fetch a URL and return text content

RULES:
- Don't narrate actions before doing them. Just make the tool call.
- Between consecutive tool calls, minimize text. Make the next tool call directly.
- ALWAYS read_file before edit_file — edit requires exact text match.
- For large files (>2000 lines), use offset/limit to read in chunks.
- Prefer batch_edit for multiple edits to the same file. Use patch_file for complex multi-hunk changes.
- Use edit_file with replace_all=true when renaming variables, functions, or classes across a file.
- Use search_files to find code patterns instead of reading entire files.
- Use get_diagnostics after making code changes to catch errors early.
- After ALL tool calls are complete, ALWAYS write a clear, helpful response summarizing your findings or explaining what you did.`
