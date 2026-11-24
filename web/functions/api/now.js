// Cloudflare Pages Function: GET /api/now
// Government AQI for Kolkata right now, from CPCB's live station feed on data.gov.in.
// Runs on Cloudflare's server nearest the visitor (inside India for Indian visitors),
// keeps the API key secret (set DATA_GOV_API_KEY in the Pages project settings),
// and caches the answer for 15 minutes so the key is not over-used.
//
// City AQI = mean over stations of each station's highest pollutant sub-index
// (CPCB method: a station needs at least 3 pollutants, one of them PM2.5 or PM10).

const FEED = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69";

export async function onRequest({ env }) {
  const headers = { "content-type": "application/json", "cache-control": "public, max-age=900" };
  if (!env.DATA_GOV_API_KEY) {
    return new Response(JSON.stringify({ error: "DATA_GOV_API_KEY not set" }), { status: 500, headers });
  }
  const q = new URLSearchParams({ "api-key": env.DATA_GOV_API_KEY, format: "json", limit: "500", "filters[city]": "Kolkata" });
  try {
    const r = await fetch(`${FEED}?${q}`, { cf: { cacheTtl: 900, cacheEverything: true },
                                            headers: { "user-agent": "kolkata-aqi-forecast" } });
    if (!r.ok) throw new Error(`data.gov.in answered ${r.status}`);
    const records = (await r.json()).records || [];
    const byStation = {};
    for (const x of records) {
      const v = parseFloat(x.avg_value);
      if (!Number.isFinite(v)) continue;
      (byStation[x.station] ||= {})[String(x.pollutant_id).toUpperCase()] = v;
    }
    const stations = Object.entries(byStation)
      .filter(([, p]) => Object.keys(p).length >= 3 && ("PM2.5" in p || "PM10" in p))
      .map(([name, p]) => ({ name: name.replace(/, Kolkata - WBPCB$/, ""), aqi: Math.max(...Object.values(p)) }));
    if (!stations.length) throw new Error("no usable Kolkata readings");
    const aqi = Math.round(stations.reduce((s, x) => s + x.aqi, 0) / stations.length);
    const feed_time = records.map(x => x.last_update || "").sort().pop();
    return new Response(JSON.stringify({ aqi, stations: stations.length, detail: stations, feed_time,
                                         fetched: new Date().toISOString(), source: "live" }), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e.message || e) }), { status: 502, headers });
  }
}
