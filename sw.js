/* Service worker.
 *
 * Two different caching needs, so two strategies:
 *
 * The shell (index.html, the manifest, icons) changes rarely and must load
 * instantly, so it is served from cache first and refreshed in the
 * background.
 *
 * The data files change every morning and a stale projection is worse than
 * a slow one, so those go to the network first and fall back to cache only
 * when the phone is offline.
 */
const VERSION = "v1";
const SHELL = `shell-${VERSION}`;
const DATA = `data-${VERSION}`;

const SHELL_FILES = [
  "./index.html",
  "./manifest.webmanifest",
  "./icon-192.png",
  "./icon-512.png",
];

const DATA_FILES = [
  "projections.json", "scorecard.json", "receptions.json",
  "receptions_proj.json", "receptions_grade.json",
  "receptions_score.json", "nfl_games.json",
  "cfb_games.json", "td_model.json", "td_grade.json",
  "nba_model.json", "nba_grade.json", "nhl_model.json", "nhl_grade.json",
];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(SHELL)
      .then(c => c.addAll(SHELL_FILES))
      .catch(() => {})            // a missing file must not block install
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== SHELL && k !== DATA).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Live scores are never cached - a stale score is worse than none.
  if (url.hostname.endsWith("espn.com")) return;

  const isData = DATA_FILES.some(f => url.pathname.endsWith(f));

  if (isData) {
    e.respondWith(
      fetch(req)
        .then(res => {
          const copy = res.clone();
          caches.open(DATA).then(c => c.put(req, copy)).catch(() => {});
          return res;
        })
        .catch(() => caches.match(req))      // offline: last known data
    );
    return;
  }

  // Shell: cache first, refresh behind the scenes.
  e.respondWith(
    caches.match(req).then(hit => {
      const live = fetch(req).then(res => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(SHELL).then(c => c.put(req, copy)).catch(() => {});
        }
        return res;
      }).catch(() => hit);
      return hit || live;
    })
  );
});
