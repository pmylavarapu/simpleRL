import { signIn, auth } from "@/lib/auth";
import { redirect } from "next/navigation";

export default async function SignInPage() {
  const session = await auth();
  if (session?.user) redirect("/");

  return (
    <div className="max-w-sm mx-auto pt-12 space-y-6">
      <div className="text-center space-y-2">
        <p className="eyebrow">Access</p>
        <h1 className="text-2xl font-medium tracking-tightest">Sign in to Echo KB</h1>
        <p className="text-[13px] text-muted">Your FSRS progress is saved to your Google account.</p>
      </div>
      <div className="sheet p-6">
        <form
          action={async () => {
            "use server";
            await signIn("google", { redirectTo: "/" });
          }}
        >
          <button
            type="submit"
            className="w-full rounded-md bg-fg text-accent-fg py-3 text-[14px] font-medium hover:opacity-90 transition-opacity"
          >
            Continue with Google
          </button>
        </form>
      </div>
      <p className="text-[11px] text-muted text-center">
        No third-party trackers. Progress stays in your account.
      </p>
    </div>
  );
}
