import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

// Notes live at /content/notes/<sectionSlug>/<topicSlug>.md
// Frontmatter is optional. Absent file = "not written yet".

const NOTES_DIR = path.join(process.cwd(), "content", "notes");

export type Note = {
  content: string;
  frontmatter: Record<string, unknown>;
  exists: boolean;
};

export function loadNote(sectionSlug: string, topicSlug: string): Note {
  const file = path.join(NOTES_DIR, sectionSlug, `${topicSlug}.md`);
  if (!fs.existsSync(file)) {
    return { content: "", frontmatter: {}, exists: false };
  }
  const raw = fs.readFileSync(file, "utf8");
  const parsed = matter(raw);
  return {
    content: parsed.content,
    frontmatter: parsed.data,
    exists: true,
  };
}
