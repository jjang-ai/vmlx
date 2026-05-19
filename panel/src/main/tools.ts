import { z } from 'zod';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

const allowedDirs = [process.cwd(), os.tmpdir(), os.homedir()];

function isPathAllowed(targetPath: string): boolean {
  const resolved = path.resolve(path.normalize(targetPath));
  return allowedDirs.some((dir) => resolved.startsWith(dir));
}

export const tools = {
  readFile: {
    description: 'Read the contents of a file',
    parameters: z.object({
      path: z.string().describe('The path to the file to read'),
    }),
    execute: async ({ path: filePath }: { path: string }) => {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        return content;
      } catch (error) {
        return `Error reading file: ${error}`;
      }
    },
  },

  writeFile: {
    description: 'Write content to a file',
    parameters: z.object({
      path: z.string().describe('The path to the file to write'),
      content: z.string().describe('The content to write to the file'),
    }),
    execute: async ({ path: filePath, content }: { path: string; content: string }) => {
      try {
        if (!isPathAllowed(filePath)) {
          return `Error: Path "${filePath}" is not within allowed directories.`;
        }
        fs.writeFileSync(filePath, content, 'utf-8');
        return `File written successfully to ${filePath}`;
      } catch (error) {
        return `Error writing file: ${error}`;
      }
    },
  },

  editFile: {
    description: 'Edit a file by replacing old content with new content',
    parameters: z.object({
      path: z.string().describe('The path to the file to edit'),
      oldContent: z.string().describe('The content to replace'),
      newContent: z.string().describe('The new content to insert'),
    }),
    execute: async ({ path: filePath, oldContent, newContent }: { path: string; oldContent: string; newContent: string }) => {
      try {
        if (!isPathAllowed(filePath)) {
          return `Error: Path "${filePath}" is not within allowed directories.`;
        }
        const content = fs.readFileSync(filePath, 'utf-8');
        const newFileContent = content.replace(oldContent, newContent);
        fs.writeFileSync(filePath, newFileContent, 'utf-8');
        return `File edited successfully at ${filePath}`;
      } catch (error) {
        return `Error editing file: ${error}`;
      }
    },
  },

  runCommand: {
    description: 'Run a shell command',
    parameters: z.object({
      command: z.string().describe('The command to run'),
    }),
    execute: async ({ command }: { command: string }) => {
      try {
        const output = execSync(command, { encoding: 'utf-8' });
        return output;
      } catch (error) {
        return `Error running command: ${error}`;
      }
    },
  },
};