import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { schedule, type Grade } from "@/lib/fsrs";
import { getCard } from "@/lib/cards";

const BodySchema = z.object({
  cardId: z.string().min(1),
  grade: z.enum(["again", "hard", "good", "easy"]),
});

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const userId = (session.user as { id?: string }).id;
  if (!userId) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const body = BodySchema.parse(await req.json());
  const card = getCard(body.cardId);
  if (!card) {
    return NextResponse.json({ error: "unknown card" }, { status: 404 });
  }

  const existing = await prisma.reviewState.findUnique({
    where: { userId_cardId: { userId, cardId: body.cardId } },
  });

  const update = schedule(existing, body.grade as Grade);

  const [saved] = await prisma.$transaction([
    prisma.reviewState.upsert({
      where: { userId_cardId: { userId, cardId: body.cardId } },
      create: { userId, cardId: body.cardId, ...update },
      update,
    }),
    prisma.reviewEvent.create({
      data: { userId, cardId: body.cardId, grade: body.grade },
    }),
  ]);

  return NextResponse.json({ ok: true, due: saved.due, state: saved.state });
}
