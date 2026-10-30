const themeToggle = document.getElementById('themeToggle');
const htmlElement = document.documentElement;

const currentTheme = localStorage.getItem('theme') || 'light';
htmlElement.setAttribute('data-theme', currentTheme);

themeToggle.addEventListener('click', () => {
    const newTheme = htmlElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    htmlElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
});


// ============================================================
// TAB MANAGEMENT
// ============================================================

const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(btn => {

    btn.addEventListener('click', () => {

        const tabName = btn.getAttribute('data-tab');

        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        tabContents.forEach(content => content.classList.remove('active'));

        const selectedTab = document.getElementById(`${tabName}-tab`);

        if (selectedTab) {
            selectedTab.classList.add('active');
        }


        // Dashboard
        if (tabName === 'dashboard') {
            loadDashboard();
        }


        // Insights
        if (tabName === 'insights') {
            loadInsights();
        }


        // History
        if (tabName === 'history') {
            loadHistory();
        }

    });

});


// ============================================================
// FORM HANDLING
// ============================================================

const form = document.getElementById('assessmentForm');
const stressButtons = document.querySelectorAll('.stress-btn');
const stressInput = document.getElementById('stress');
const submitButton = form.querySelector('button[type="submit"]');


stressButtons.forEach(btn => {

    btn.addEventListener('click', (e) => {

        e.preventDefault();

        stressButtons.forEach(b => b.classList.remove('active'));

        btn.classList.add('active');

        stressInput.value = btn.getAttribute('data-stress');

    });

});


form.addEventListener('submit', async (e) => {

    e.preventDefault();

    const formData = {

        age: parseInt(document.getElementById('age').value),

        gender: document.getElementById('gender').value,

        country: document.getElementById('country').value,

        academic_level:
            document.getElementById('academicLevel').value,

        most_used_platform:
            document.getElementById('platform').value,

        purpose_of_use:
            document.getElementById('purpose').value,

        avg_daily_usage_hours:
            parseFloat(document.getElementById('screenTime').value),

        daily_unlocks:
            parseInt(document.getElementById('unlocks').value),

        study_hours:
            parseFloat(document.getElementById('studyHours').value),

        physical_activity_hours:
            parseFloat(document.getElementById('activity').value),

        sleep_hours_per_night:
            parseFloat(document.getElementById('sleep').value),

        stress_level:
            stressInput.value
    };


    submitButton.classList.add('loading');

    showLoader(true);


    try {

        const response = await fetch('/api/predict', {

            method: 'POST',

            headers: {
                'Content-Type': 'application/json'
            },

            body: JSON.stringify(formData)

        });


        if (!response.ok) {

            throw new Error('Prediction failed');

        }


        const result = await response.json();


        displayResults(result);

        saveToHistory(result);

        showDashboard();


    } catch (error) {

        console.error('Prediction error:', error);

        alert('Error: ' + error.message);

    } finally {

        submitButton.classList.remove('loading');

        showLoader(false);

    }

});


// ============================================================
// RESULTS DISPLAY
// ============================================================

function displayResults(result) {

    const resultsSection =
        document.getElementById('resultsSection');

    resultsSection.style.display = 'block';


    // Score
    document.getElementById('scoreNumber').textContent =
        Number(result.score).toFixed(2);


    document.getElementById('scoreBand').textContent =
        result.band;


    document.getElementById('scoreBand').style.color =
        result.band_color;


    document.getElementById('percentileInfo').textContent =
        `${result.percentile}th percentile`;


    // Score Gauge
    drawScoreGauge(
        result.score,
        result.band_color
    );


    // Wellness
    const mhPercent =
        (result.score / 10) * 100;

    const wellnessPercent =
        result.wellness_score;


    document.getElementById('mhBar').style.width =
        mhPercent + '%';

    document.getElementById('mhValue').textContent =
        Math.round(mhPercent) + '%';


    document.getElementById('wellnessBar').style.width =
        wellnessPercent + '%';

    document.getElementById('wellnessValue').textContent =
        Math.round(wellnessPercent) + '%';


    // Risk
    const riskAlert =
        document.getElementById('riskAlert');


    riskAlert.innerHTML =
        result.risk_assessment;


    riskAlert.className =
        'risk-alert';


    if (result.score < 3) {

        riskAlert.classList.add('critical');

    } else if (result.score < 5) {

        riskAlert.classList.add('high');

    } else {

        riskAlert.classList.add('low');

    }


    // Factors
    displayFactors(result.explanation);


    // Recommendations
    displayRecommendations(result.recommendations);


    // Summary
    document.getElementById('summaryBox').innerHTML =
        result.summary;


    // Scroll
    resultsSection.scrollIntoView({
        behavior: 'smooth'
    });

}


// ============================================================
// SCORE GAUGE
// ============================================================

function drawScoreGauge(score, color) {

    const canvas =
        document.getElementById('scoreGauge');

    const ctx =
        canvas.getContext('2d');


    const radius = 80;

    const centerX =
        canvas.width / 2;

    const centerY =
        canvas.height / 2;


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    // Background circle
    ctx.beginPath();

    ctx.arc(
        centerX,
        centerY,
        radius,
        0,
        Math.PI * 2
    );

    ctx.strokeStyle = '#e5e7eb';

    ctx.lineWidth = 12;

    ctx.stroke();


    // Progress
    const startAngle =
        -Math.PI / 2;

    const endAngle =
        startAngle +
        (score / 10) * Math.PI * 2;


    ctx.beginPath();

    ctx.arc(
        centerX,
        centerY,
        radius,
        startAngle,
        endAngle
    );

    ctx.strokeStyle = color;

    ctx.lineWidth = 12;

    ctx.lineCap = 'round';

    ctx.stroke();

}


// ============================================================
// FACTORS
// ============================================================

function displayFactors(factors) {

    const container =
        document.getElementById('factorsBreakdown');


    container.innerHTML = '';


    factors.forEach(factor => {

        const card =
            document.createElement('div');


        card.className =
            `factor-item ${factor.category}`;


        card.innerHTML = `

            <div class="factor-name">
                ${factor.factor}
                (${factor.impact_score > 0 ? '+' : ''}
                ${factor.impact_score})
            </div>

            <div class="factor-explanation">
                ${factor.explanation}
            </div>

            <div class="factor-recommendation">
                💡 ${factor.recommendation}
            </div>

        `;


        container.appendChild(card);

    });

}


// ============================================================
// RECOMMENDATIONS
// ============================================================

function displayRecommendations(recommendations) {

    const container =
        document.getElementById('recommendations');


    container.innerHTML = '';


    recommendations.forEach(rec => {

        const card =
            document.createElement('div');


        card.className =
            `recommendation-card ${rec.priority}`;


        const items =
            rec.action_items
                .map(item => `<li>${item}</li>`)
                .join('');


        card.innerHTML = `

            <div class="rec-title">
                ${rec.title}
            </div>

            <div class="rec-description">
                ${rec.description}
            </div>

            <ul class="rec-items">
                ${items}
            </ul>

        `;


        container.appendChild(card);

    });

}


// ============================================================
// CHART MANAGEMENT
// ============================================================

// IMPORTANT:
// Chart.js does not allow creating another chart on the same
// canvas without destroying the old one first.

const chartInstances = {};


function destroyChart(chartId) {

    if (chartInstances[chartId]) {

        try {

            chartInstances[chartId].destroy();

        } catch (error) {

            console.warn(
                `Could not destroy ${chartId}:`,
                error
            );

        }

        chartInstances[chartId] = null;

    }


    const canvas =
        document.getElementById(chartId);


    if (canvas) {

        const ctx =
            canvas.getContext('2d');


        if (ctx) {

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

        }

    }

}


// ============================================================
// DASHBOARD
// ============================================================

async function loadDashboard() {

    try {

        const response =
            await fetch('/api/dashboard', {
                cache: 'no-store'
            });


        if (!response.ok) {

            throw new Error(
                `Dashboard request failed (${response.status})`
            );

        }


        const data =
            await response.json();


        // Metrics
        const metricsRow =
            document.getElementById('metricsRow');


        if (metricsRow) {

            metricsRow.innerHTML = '';


            data.metrics.forEach(metric => {

                const box =
                    document.createElement('div');


                box.className =
                    'metric-box';


                box.innerHTML = `

                    <div class="metric-box-label">
                        ${metric.label}
                    </div>

                    <div class="metric-box-value">
                        ${metric.value}
                    </div>

                    <div class="metric-box-change
                        ${metric.change.includes('-')
                            ? 'negative'
                            : ''}">
                        ${metric.change}
                    </div>

                `;


                metricsRow.appendChild(box);

            });

        }


        // Charts
        if (document.getElementById('trendChart')) {

            drawTrendChart(data.trends);

        }


        if (document.getElementById('distributionChart')) {

            drawDistributionChart();

        }


        if (document.getElementById('modelChart')) {

            drawModelChart();

        }


    } catch (error) {

        console.error(
            'Dashboard error:',
            error
        );

    }

}


// ============================================================
// TREND CHART
// ============================================================

function drawTrendChart(trends) {

    const canvas =
        document.getElementById('trendChart');


    if (!canvas) return;


    destroyChart('trendChart');


    const ctx =
        canvas.getContext('2d');


    chartInstances.trendChart =
        new Chart(ctx, {

            type: 'line',

            data: {

                labels:
                    trends.map(t => t.date),

                datasets: [{

                    label:
                        'Mental Health Score',

                    data:
                        trends.map(t => t.score),

                    borderColor:
                        '#7c3aed',

                    backgroundColor:
                        'rgba(124, 58, 237, 0.1)',

                    borderWidth: 3,

                    fill: true,

                    tension: 0.4,

                    pointRadius: 6,

                    pointBackgroundColor:
                        '#7c3aed',

                    pointBorderColor:
                        '#fff',

                    pointBorderWidth: 2

                }]

            },

            options: {

                responsive: true,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    y: {
                        min: 0,
                        max: 10
                    }

                }

            }

        });

}


// ============================================================
// DISTRIBUTION CHART
// ============================================================

function drawDistributionChart() {

    const canvas =
        document.getElementById('distributionChart');


    if (!canvas) return;


    destroyChart('distributionChart');


    const ctx =
        canvas.getContext('2d');


    chartInstances.distributionChart =
        new Chart(ctx, {

            type: 'doughnut',

            data: {

                labels: [

                    'Critical',

                    'At Risk',

                    'Struggling',

                    'Fair',

                    'Good',

                    'Thriving'

                ],

                datasets: [{

                    data: [
                        5,
                        12,
                        18,
                        25,
                        28,
                        12
                    ],

                    backgroundColor: [

                        '#7f1d1d',

                        '#dc2626',

                        '#f59e0b',

                        '#fbbf24',

                        '#3b82f6',

                        '#10b981'

                    ],

                    borderWidth: 2,

                    borderColor: '#fff'

                }]

            },

            options: {

                responsive: true,

                plugins: {

                    legend: {
                        position: 'bottom'
                    }

                }

            }

        });

}


// ============================================================
// MODEL CHART
// ============================================================

function drawModelChart() {

    const canvas =
        document.getElementById('modelChart');


    if (!canvas) return;


    destroyChart('modelChart');


    const ctx =
        canvas.getContext('2d');


    chartInstances.modelChart =
        new Chart(ctx, {

            type: 'radar',

            data: {

                labels: [

                    'R² Score',

                    'Speed',

                    'Interpretability',

                    'Robustness',

                    'Accuracy'

                ],

                datasets: [

                    {

                        label:
                            'Selected Model (XGBoost)',

                        data: [

                            0.92,

                            0.88,

                            0.70,

                            0.95,

                            0.93

                        ],

                        borderColor:
                            '#7c3aed',

                        backgroundColor:
                            'rgba(124, 58, 237, 0.2)',

                        borderWidth: 2

                    },

                    {

                        label:
                            'Baseline',

                        data: [

                            0.74,

                            0.65,

                            0.85,

                            0.70,

                            0.72

                        ],

                        borderColor:
                            '#06b6d4',

                        backgroundColor:
                            'rgba(6, 182, 212, 0.2)',

                        borderWidth: 2

                    }

                ]

            },

            options: {

                responsive: true,

                plugins: {

                    legend: {
                        position: 'bottom'
                    }

                },

                scales: {

                    r: {

                        min: 0,

                        max: 1

                    }

                }

            }

        });

}


// ============================================================
// INSIGHTS
// ============================================================

async function loadInsights() {

    const featureList =
        document.getElementById('featureList');


    const statsList =
        document.getElementById('statsList');


    if (!featureList || !statsList) {

        return;

    }


    featureList.innerHTML =
        '<p style="color: var(--text-secondary);">Loading insights...</p>';


    statsList.innerHTML =
        '<p style="color: var(--text-secondary);">Loading statistics...</p>';


    try {

        const response =
            await fetch('/api/stats', {
                cache: 'no-store'
            });


        if (!response.ok) {

            throw new Error(
                `Insights request failed (${response.status})`
            );

        }


        const stats =
            await response.json();


        // ----------------------------------------------------
        // FEATURE IMPORTANCE
        // ----------------------------------------------------

        const featureImportances =
            stats?.model?.feature_importances || [];


        if (
            !Array.isArray(featureImportances) ||
            featureImportances.length === 0
        ) {

            featureList.innerHTML =
                '<p style="color: var(--text-secondary);">Feature importance data is not available.</p>';

        } else {

            const items =
                featureImportances

                    .map(item => {

                        const name =
                            Array.isArray(item)
                                ? item[0]
                                : item.name;


                        const rawValue =
                            Array.isArray(item)
                                ? item[1]
                                : item.value;


                        const value =
                            Number(rawValue);


                        return {

                            name:
                                name ?? 'Unknown feature',

                            value:
                                Number.isFinite(value)
                                    ? value
                                    : 0

                        };

                    })

                    .sort(
                        (a, b) =>
                            b.value - a.value
                    );


            const maxValue =
                Math.max(
                    ...items.map(
                        item =>
                            Math.abs(item.value)
                    ),
                    1
                );


            featureList.innerHTML =
                items.map(item => {

                    const width =
                        Math.max(
                            3,
                            Math.min(
                                100,
                                (
                                    Math.abs(item.value) /
                                    maxValue
                                ) * 100
                            )
                        );


                    return `

                        <div
                            class="feature-item"
                            style="margin-bottom:0.9rem;"
                        >

                            <div
                                style="
                                display:flex;
                                justify-content:space-between;
                                gap:1rem;
                                margin-bottom:0.35rem;
                                "
                            >

                                <strong>
                                    ${escapeHtml(
                                        String(item.name)
                                    )}
                                </strong>

                                <span>
                                    ${formatInsightNumber(
                                        item.value
                                    )}
                                </span>

                            </div>


                            <div
                                style="
                                height:8px;
                                background:var(--border-color);
                                border-radius:999px;
                                overflow:hidden;
                                "
                            >

                                <div
                                    style="
                                    height:100%;
                                    width:${width}%;
                                    background:var(--primary-color, #7c3aed);
                                    border-radius:999px;
                                    "
                                ></div>

                            </div>

                        </div>

                    `;

                }).join('');

        }


        // ----------------------------------------------------
        // KEY STATISTICS
        // ----------------------------------------------------

        const model =
            stats?.model || {};


        const dataset =
            stats?.dataset || {};


        const performance =
            model.performance || {};


        const baseline =
            dataset.baseline || {};


        const distribution =
            dataset.distribution || {};


        const statRows = [];


        statRows.push([

            'Model',

            model.name ||
                'Mental Health Model'

        ]);


        statRows.push([

            'Dataset Records',

            dataset.total_records ??
                '—'

        ]);


        Object.entries(
            performance
        ).forEach(
            ([key, value]) => {

                statRows.push([

                    formatInsightLabel(key),

                    formatInsightValue(value)

                ]);

            }
        );


        Object.entries(
            baseline
        ).forEach(
            ([key, value]) => {

                statRows.push([

                    formatInsightLabel(key),

                    formatInsightValue(value)

                ]);

            }
        );


        Object.entries(
            distribution
        ).forEach(
            ([key, value]) => {

                statRows.push([

                    `Distribution: ${formatInsightLabel(key)}`,

                    formatInsightValue(value)

                ]);

            }
        );


        statRows.push([

            'Updated',

            stats.timestamp
                ? new Date(
                    stats.timestamp
                  ).toLocaleString()
                : '—'

        ]);


        statsList.innerHTML =
            statRows.map(
                ([label, value]) => `

                    <div
                        class="stat-item"
                        style="
                        display:flex;
                        justify-content:space-between;
                        gap:1rem;
                        padding:0.7rem 0;
                        border-bottom:1px solid var(--border-color);
                        "
                    >

                        <span
                            style="
                            color:var(--text-secondary);
                            "
                        >
                            ${escapeHtml(
                                String(label)
                            )}
                        </span>


                        <strong
                            style="
                            text-align:right;
                            "
                        >
                            ${escapeHtml(
                                String(value)
                            )}
                        </strong>

                    </div>

                `
            ).join('');


    } catch (error) {

        console.error(
            'Insights error:',
            error
        );


        featureList.innerHTML =
            '<p style="color:var(--text-secondary);">Unable to load feature insights.</p>';


        statsList.innerHTML =
            '<p style="color:var(--text-secondary);">Unable to load statistics. Please refresh the page.</p>';

    }

}


// ============================================================
// INSIGHTS HELPERS
// ============================================================

function formatInsightLabel(key) {

    return String(key)

        .replace(/_/g, ' ')

        .replace(
            /\b\w/g,
            char =>
                char.toUpperCase()
        );

}


function formatInsightNumber(value) {

    const number =
        Number(value);


    if (!Number.isFinite(number)) {

        return String(
            value ?? '—'
        );

    }


    if (Math.abs(number) >= 1) {

        return number.toFixed(2);

    }


    return number.toFixed(4);

}


function formatInsightValue(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return '—';

    }


    if (typeof value === 'number') {

        return formatInsightNumber(
            value
        );

    }


    if (typeof value === 'object') {

        return Object.entries(value)

            .map(
                ([key, item]) =>
                    `${formatInsightLabel(key)}: ${formatInsightValue(item)}`
            )

            .join(' • ');

    }


    return String(value);

}


function escapeHtml(value) {

    return String(value)

        .replace(
            /&/g,
            '&amp;'
        )

        .replace(
            /</g,
            '&lt;'
        )

        .replace(
            />/g,
            '&gt;'
        )

        .replace(
            /"/g,
            '&quot;'
        )

        .replace(
            /'/g,
            '&#039;'
        );

}


// ============================================================
// HISTORY MANAGEMENT
// ============================================================

function saveToHistory(result) {

    let history =
        JSON.parse(
            localStorage.getItem(
                'mhHistory'
            ) || '[]'
        );


    history.push({

        timestamp:
            new Date().toLocaleString(),

        score:
            result.score,

        band:
            result.band,

        wellness:
            result.wellness_score

    });


    history =
        history.slice(-30);


    localStorage.setItem(
        'mhHistory',
        JSON.stringify(history)
    );


    loadHistory();

}


function loadHistory() {

    const history =
        JSON.parse(
            localStorage.getItem(
                'mhHistory'
            ) || '[]'
        );


    const container =
        document.getElementById(
            'historyList'
        );


    if (!container) {

        return;

    }


    if (history.length === 0) {

        container.innerHTML =
            '<p style="color: var(--text-secondary);">No history yet</p>';


        destroyChart(
            'historyChart'
        );


        return;

    }


    container.innerHTML = '';


    // Do not modify original history array.
    const displayHistory =
        [...history].reverse();


    displayHistory.forEach(
        (entry) => {

            const item =
                document.createElement(
                    'div'
                );


            item.style.cssText =
                'padding: 0.75rem; border-bottom: 1px solid var(--border-color);';


            item.innerHTML = `

                <strong>
                    ${entry.score}/10
                </strong>

                (${entry.band})

                -

                ${entry.timestamp}

            `;


            container.appendChild(
                item
            );

        }
    );


    // --------------------------------------------------------
    // HISTORY CHART
    // --------------------------------------------------------

    if (displayHistory.length > 1) {

        destroyChart(
            'historyChart'
        );


        const canvas =
            document.getElementById(
                'historyChart'
            );


        if (!canvas) {

            return;

        }


        const ctx =
            canvas.getContext('2d');


        chartInstances.historyChart =
            new Chart(
                ctx,
                {

                    type: 'line',

                    data: {

                        labels:
                            displayHistory.map(
                                h =>
                                    h.timestamp.split(
                                        ' '
                                    )[0]
                            ),

                        datasets: [{

                            label:
                                'Score History',

                            data:
                                displayHistory.map(
                                    h =>
                                        h.score
                                ),

                            borderColor:
                                '#7c3aed',

                            backgroundColor:
                                'rgba(124, 58, 237, 0.1)',

                            borderWidth: 2,

                            fill: true,

                            tension: 0.4

                        }]

                    },

                    options: {

                        responsive: true,

                        plugins: {

                            legend: {
                                display: false
                            }

                        },

                        scales: {

                            y: {

                                min: 0,

                                max: 10

                            }

                        }

                    }

                }
            );

    } else {

        destroyChart(
            'historyChart'
        );

    }

}


// ============================================================
// UTILITIES
// ============================================================

function showLoader(show) {

    const loader =
        document.getElementById(
            'loader'
        );


    if (loader) {

        loader.classList.toggle(
            'show',
            show
        );

    }

}


function showDashboard() {

    const dashboardTab =
        document.querySelector(
            '.tab-btn[data-tab="dashboard"]'
        );


    if (dashboardTab) {

        dashboardTab.click();

    }

}


// ============================================================
// MODEL INFO
// ============================================================

async function loadModelInfo() {

    try {

        const response =
            await fetch(
                '/api/stats',
                {
                    cache: 'no-store'
                }
            );


        if (!response.ok) {

            throw new Error(
                `Model info request failed (${response.status})`
            );

        }


        const stats =
            await response.json();


        const modelBadge =
            document.getElementById(
                'modelBadge'
            );


        if (modelBadge) {

            modelBadge.textContent =
                `Model: ${stats.model.name}`;

        }


    } catch (error) {

        console.error(
            'Model load error:',
            error
        );

    }

}


// ============================================================
// EXPORT FUNCTIONALITY
// ============================================================

const exportBtn =
    document.getElementById(
        'exportBtn'
    );


if (exportBtn) {

    exportBtn.addEventListener(
        'click',
        () => {

            const element =
                document.getElementById(
                    'resultsSection'
                );


            const opt = {

                margin: 10,

                filename:
                    `mental-health-report-${new Date().toISOString().split('T')[0]}.pdf`,

                image: {

                    type: 'jpeg',

                    quality: 0.98

                },

                html2canvas: {

                    scale: 2

                },

                jsPDF: {

                    orientation:
                        'portrait',

                    unit: 'mm',

                    format: 'a4'

                }

            };


            html2pdf()
                .set(opt)
                .from(element)
                .save();

        }
    );

}


// ============================================================
// INITIALIZATION
// ============================================================

window.addEventListener(
    'load',
    () => {

        loadModelInfo();

        loadHistory();

        // Preload insights.
        loadInsights();

    }
);