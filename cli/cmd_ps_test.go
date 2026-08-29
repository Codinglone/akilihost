package cli

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCheckHealthResultOK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"id": "test-model"}},
		})
	}))
	defer ts.Close()

	got := checkHealthResult(ts.URL + "/v1/models")
	if !strings.Contains(got, "ok") {
		t.Errorf("expected health ok, got: %s", got)
	}
	if !strings.Contains(got, "test-model") {
		t.Errorf("expected model id in output, got: %s", got)
	}
}

func TestCheckHealthResultUnreachable(t *testing.T) {
	got := checkHealthResult("http://127.0.0.1:1/v1/models")
	if !strings.Contains(got, "unreachable") && !strings.Contains(got, "timeout") {
		t.Errorf("expected unreachable/timeout, got: %s", got)
	}
}

func TestCheckHealthResultNoModels(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data": []}`)
	}))
	defer ts.Close()

	got := checkHealthResult(ts.URL + "/v1/models")
	if strings.Contains(got, "ok (model:") {
		t.Errorf("should not report ok with empty data, got: %s", got)
	}
}
