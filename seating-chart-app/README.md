# Seating Chart App

A simple, elegant web app that helps event guests find their assigned table. Perfect for weddings, galas, and corporate events.

**Guest side:** Guests type their name and instantly see their table number.
**Admin side:** Upload an Excel file with guest names and table assignments.

---

## Quick Start (Local Development)

### Prerequisites

- [Node.js](https://nodejs.org/) version 18 or later

### 1. Install dependencies

```bash
# From the project root
cd backend && npm install
cd ../frontend && npm install
```

### 2. Start the backend

```bash
cd backend
npm start
```

The API will run at `http://localhost:5000`.

### 3. Start the frontend (in a second terminal)

```bash
cd frontend
npm start
```

The app will open at `http://localhost:3000`.

---

## How to Use

### Admin Panel

1. Open the app and click the **Admin** tab.
2. Enter the admin password: `seating2024`
3. Upload your Excel file (`.xlsx`, `.xls`, or `.csv`).
4. Your guest list is now live — guests can search for their names.

### Excel File Format

Your spreadsheet needs exactly two columns named **Name** and **Table**:

| Name          | Table |
| ------------- | ----- |
| Jane Smith    | 1     |
| John Doe      | 3     |
| Maria Garcia  | 2     |

- Column headers are case-insensitive (`name`, `Name`, `NAME` all work).
- Extra columns are ignored silently.
- Blank rows are skipped.

### QR Code

After logging in as admin, a **QR Code** tab appears. It generates a QR code pointing to your app's URL. You can:

- Download it as a PNG image
- Print it and place it at your event entrance

---

## Deploy to Vercel (Free)

### Backend

1. Push this repo to GitHub.
2. Go to [vercel.com](https://vercel.com) and sign up / log in.
3. Click **Add New → Project** and import your repo.
4. Set the **Root Directory** to `backend`.
5. Vercel will auto-detect the configuration from `vercel.json`.
6. Click **Deploy**. Note the URL (e.g., `https://seating-backend.vercel.app`).

### Frontend

1. In Vercel, create another project from the same repo.
2. Set the **Root Directory** to `frontend`.
3. Under **Environment Variables**, add:
   - `REACT_APP_API_URL` = your backend Vercel URL (e.g., `https://seating-backend.vercel.app`)
4. Click **Deploy**. The frontend URL is what guests will visit.

After deploying, open the app, go to the QR Code tab, and download/print the QR code that points to your live frontend URL.

---

## Customisation

### Change the event name

Edit `frontend/src/App.jsx` and update these two lines near the top:

```js
const EVENT_TITLE = "Welcome to Our Wedding!";
const EVENT_SUBTITLE = "Search your name to find your table";
```

### Change the admin password

Edit `backend/server.js` and update:

```js
const ADMIN_PASSWORD = "seating2024";
```

### Change the colours

Edit `frontend/src/App.css`. The main gradient appears in several places:

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Replace `#667eea` and `#764ba2` with your preferred colours.

---

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| "Connection error" when searching | Make sure the backend is running. Check the `REACT_APP_API_URL` in `frontend/.env`. |
| Upload says "No file uploaded" | Make sure you selected a file before clicking Upload. |
| "Could not find required columns" | Your Excel file must have columns named exactly **Name** and **Table**. |
| Guest search returns no results | The guest list may not have been uploaded yet. Check the Admin tab. |
| App looks broken on mobile | Clear your browser cache and reload. |
| Backend won't start | Run `cd backend && npm install` to make sure dependencies are installed. |

---

## Tech Stack

- **Backend:** Node.js, Express, multer, xlsx
- **Frontend:** React, qrcode.react
- **Storage:** Local JSON file (no database needed)
- **Deployment:** Vercel (free tier)
