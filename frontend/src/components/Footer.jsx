import React from "react";

export default function Footer() {
  return (
    <footer className="footer bg-dark text-white py-4">
      <div className="container text-center">
        <p className="mb-0">
          &copy; {new Date().getFullYear()} Nuu Mobile. All Rights Reserved.
        </p>
        <p className="mb-0">Made by the NuuB Team</p>
        <div className="mt-2">
          <a
            href="/privacy-policy"
            className="text-white mx-2 text-decoration-none"
            aria-label="Privacy Policy"
          >
            Privacy Policy
          </a>
          <a
            href="/terms-of-service"
            className="text-white mx-2 text-decoration-none"
            aria-label="Terms of Service"
          >
            Terms of Service
          </a>
          <a
            href="/contact"
            className="text-white mx-2 text-decoration-none"
            aria-label="Contact Us"
          >
            Contact Us
          </a>
        </div>
      </div>
    </footer>
  );
}
