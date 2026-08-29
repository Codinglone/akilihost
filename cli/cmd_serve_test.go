package cli

import (
	"net"
	"testing"
)

func TestResolvePortExplicitFree(t *testing.T) {
	listener, _ := net.Listen("tcp", ":0")
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	got, err := resolvePort(port, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != port {
		t.Errorf("resolvePort(%d, true) = %d, want %d", port, got, port)
	}
}

func TestResolvePortExplicitBusy(t *testing.T) {
	listener, _ := net.Listen("tcp", ":0")
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port

	_, err := resolvePort(port, true)
	if err == nil {
		t.Fatal("expected error for busy explicit port, got nil")
	}
}

func TestResolvePortAutoIncrement(t *testing.T) {
	listener, _ := net.Listen("tcp", ":0")
	defer listener.Close()
	busyPort := listener.Addr().(*net.TCPAddr).Port

	got, err := resolvePort(busyPort, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == busyPort {
		t.Fatalf("expected auto-increment past busy port %d, got %d", busyPort, got)
	}
	if got < busyPort {
		t.Errorf("expected port >= %d, got %d", busyPort, got)
	}
}

func TestResolvePortDefaultFree(t *testing.T) {
	listener, err := net.Listen("tcp", ":18002")
	if err != nil {
		t.Skip("port 18002 not available, skipping")
	}
	listener.Close()

	got, err := resolvePort(18002, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 18002 {
		t.Errorf("resolvePort(18002, false) = %d, want 18002", got)
	}
}
