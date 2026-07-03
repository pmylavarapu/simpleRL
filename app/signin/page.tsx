import { signIn, auth } from "@/lib/auth";
import { redirect } from "next/navigation";

export default async function SignInPage() {
  const session = await auth();
  if (session?.user) redirect("/");

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-sm w-full border border-border rounded-lg p-6 space-y-4">
        <h1 className="text-xl font-semibold">Sign in to ASE Echo KB</h1>
        <p className="text-sm text-muted">
          Your review progress is saved to your Google account.
        </p>
        <form
          action={async () => {
            "use server";
            await signIn("google", { redirectTo: "/" });
          }}
        >
          <button
            type="submit"
            className="w-full rounded-md bg-accent text-white py-2 font-medium hover:opacity-90"
          >
            Continue with Google
          </button>
        </form>
      </div>
    </main>
  );
}
