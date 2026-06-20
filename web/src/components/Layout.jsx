import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import ChatWidget from "./ChatWidget";
import "./Layout.css";

export default function Layout() {
  return (
    <div className="app-layout">
      <Navbar />
      <main>
        <Outlet />
      </main>
      <ChatWidget />
    </div>
  );
}
