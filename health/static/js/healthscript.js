const feedUrl = "https://www.who.int/rss-feeds/news-english.xml"; // Updated URL

async function fetchRSS() {
    try {
        const response = await fetch(feedUrl);
        const data = await response.text();

        // Parse the RSS data
        const parser = new DOMParser();
        const xml = parser.parseFromString(data, "application/xml");
        const items = xml.querySelectorAll("item");

        const alertsContainer = document.getElementById("health-alerts");
        let alertsHTML = "<ul>";

        items.forEach((item, index) => {
            if (index < 5) { // Limit to 5 alerts
                const title = item.querySelector("title").textContent;
                const link = item.querySelector("link").textContent;
                alertsHTML += `<li><a href="${link}" target="_blank">${title}</a></li>`;
            }
        });

        alertsHTML += "</ul>";
        alertsContainer.innerHTML = alertsHTML;
    } catch (error) {
        console.error("Error fetching RSS feed:", error);
        document.getElementById("health-alerts").textContent =
            "Failed to load health alerts. Please try again later.";
    }
}

fetchRSS();
