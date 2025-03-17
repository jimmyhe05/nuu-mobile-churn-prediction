import React from "react";
import { Navbar, Nav, Container } from "react-bootstrap";
import logo from "../assets/nuu-logo.svg";

const Header = () => {
  const linkStyle = {
    color: "white",
    transition: "color 0.3s ease",
  };

  const handleHover = (e) => {
    e.target.style.color = "#c8d934";
  };

  const handleLeave = (e) => {
    e.target.style.color = "white";
  };

  return (
    <Navbar bg="dark" variant="dark" expand="md">
      <Container fluid>
        <Navbar.Brand href="/" className="d-flex align-items-center">
          <img
            src={logo}
            style={{
              width: "50px",
              height: "30px",
              borderRadius: "15%",
            }}
            alt="Nuu Mobile Logo"
          />
        </Navbar.Brand>

        <Navbar.Toggle
          style={{
            border: "none",
            backgroundColor: "#228B22",
            boxShadow: "none",
          }}
        ></Navbar.Toggle>

        <Navbar.Collapse>
          <Nav className="ms-auto">
            <Nav.Link
              href="/Training-for-engineers"
              style={linkStyle}
              onMouseEnter={handleHover}
              onMouseLeave={handleLeave}
            >
              Training For Engineers
            </Nav.Link>
            <Nav.Link
              href="/"
              style={linkStyle}
              onMouseEnter={handleHover}
              onMouseLeave={handleLeave}
            >
              Dashboard
            </Nav.Link>
            {/* <Nav.Link
              href="/predictions"
              style={linkStyle}
              onMouseEnter={handleHover}
              onMouseLeave={handleLeave}
            >
              Predictions
            </Nav.Link> */}
            {/* <Nav.Link
              href="/settings"
              style={linkStyle}
              onMouseEnter={handleHover}
              onMouseLeave={handleLeave}
            >
              Settings
            </Nav.Link> */}
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default Header;
