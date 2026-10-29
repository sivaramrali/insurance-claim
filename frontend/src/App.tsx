const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';
import { useEffect, useState } from 'react'
import './App.css'

type HealthResponse = {
  status: string
  service?: string
}

type VehicleClaim = {
  vehicle_age: number
  driver_age: number
  vehicle_value: number
  claim_amount: number
  repair_cost: number
  previous_claims: number
  policy_duration_months: number
  documents_complete: number
  police_report_available: number
  accident_severity: string
  total_cfs: number
}

type Counterfactual = {
  claim_amount: number
  repair_cost: number
  predicted_decision: string
}

type ExplanationResponse = {
  success: boolean
  original_prediction: string
  counterfactuals: Counterfactual[]
  disclaimer?: string
}

type DecisionType = 'approved' | 'rejected' | 'unknown'

const initialClaim: VehicleClaim = {
  vehicle_age: 1,
  driver_age: 21,
  vehicle_value: 1000000,
  claim_amount: 300000,
  repair_cost: 250000,
  previous_claims: 0,
  policy_duration_months: 24,
  documents_complete: 0,
  police_report_available: 0,
  accident_severity: 'minor',
  total_cfs: 3,
}

function formatAmount(value: number) {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2,
  }).format(value)
}

function getDecisionType(decision: string): DecisionType {
  const value = decision.trim().toLowerCase()

  if (
    /^(rejected|reject|denied|deny|declined|decline|not approved|not_approved|not-approved|not approve|unapproved|unaccepted)$/.test(
      value,
    )
  ) {
    return 'rejected'
  }

  if (/^(approved|approve|accepted|accept)$/.test(value)) {
    return 'approved'
  }

  return 'unknown'
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [apiError, setApiError] = useState('')
  const [activePage, setActivePage] = useState<
    'Overview' | 'New Claim' | 'Explanations'
  >('Overview')
  const [claim, setClaim] = useState<VehicleClaim>(initialClaim)
  const [explanation, setExplanation] =
    useState<ExplanationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [explanationError, setExplanationError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/v1/health`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Backend health check failed')
        }

        return (await response.json()) as HealthResponse
      })
      .then((data) => {
        setHealth(data)
        setApiError('')
      })
      .catch(() => {
        setHealth(null)
        setApiError('Unable to connect to the analysis service.')
      })
  }, [])

  function updateClaim<K extends keyof VehicleClaim>(
    field: K,
    value: VehicleClaim[K],
  ) {
    setClaim((previous) => ({ ...previous, [field]: value }))
  }

  async function handleSubmit(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()
    setExplanationError('')

    if (claim.repair_cost > claim.claim_amount) {
      setExplanationError(
        'Repair cost cannot exceed the claim amount.',
      )
      return
    }

    if (claim.claim_amount > claim.vehicle_value) {
      setExplanationError(
        'Claim amount cannot exceed vehicle value.',
      )
      return
    }

    if (
      !Number.isFinite(claim.vehicle_value) ||
      !Number.isFinite(claim.claim_amount) ||
      !Number.isFinite(claim.repair_cost) ||
      claim.vehicle_value <= 0 ||
      claim.claim_amount <= 0 ||
      claim.repair_cost < 0
    ) {
      setExplanationError(
        'Enter valid amounts. Vehicle value and claim amount must be greater than zero, and repair cost cannot be negative.',
      )
      return
    }

    setLoading(true)
    setExplanation(null)

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/vehicle/explanations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(claim),
        },
      )

      const data = await response.json()

      if (!response.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail
                  .map(
                    (item: { msg?: string }) =>
                      item.msg || 'Invalid input',
                  )
                  .join('; ')
              : 'The service could not process this claim.'

        throw new Error(detail)
      }

      if (
        data.success !== true ||
        typeof data.original_prediction !== 'string' ||
        !Array.isArray(data.counterfactuals)
      ) {
        throw new Error(
          'The service returned an unexpected response.',
        )
      }

      setExplanation(data as ExplanationResponse)
      setActivePage('Explanations')
    } catch (error) {
      setExplanationError(
        error instanceof Error
          ? error.message
          : 'An unexpected error occurred. Please try again.',
      )
    } finally {
      setLoading(false)
    }
  }

  function applyAlternative(item: Counterfactual) {
    setClaim((previous) => ({
      ...previous,
      claim_amount: item.claim_amount,
      repair_cost: item.repair_cost,
    }))

    setExplanationError('')
    setActivePage('New Claim')
  }

  const decisionType = explanation
    ? getDecisionType(explanation.original_prediction)
    : 'unknown'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button
          className="brand"
          type="button"
          onClick={() => setActivePage('Overview')}
          aria-label="Go to overview"
        >
          <span className="brand-mark">CT</span>

          <span className="brand-copy">
            <strong>Claim Transparency</strong>
            <span>Vehicle claim analysis</span>
          </span>
        </button>

        <p className="nav-label">MENU</p>

        <nav className="main-nav" aria-label="Main navigation">
          <button
            className={`nav-item ${
              activePage === 'Overview' ? 'active' : ''
            }`}
            onClick={() => setActivePage('Overview')}
            type="button"
          >
            <span className="nav-symbol" aria-hidden="true">
              ⌂
            </span>
            Overview
          </button>

          <button
            className={`nav-item ${
              activePage === 'New Claim' ? 'active' : ''
            }`}
            onClick={() => setActivePage('New Claim')}
            type="button"
          >
            <span className="nav-symbol" aria-hidden="true">
              +
            </span>
            New claim
          </button>

          <button
            className={`nav-item ${
              activePage === 'Explanations' ? 'active' : ''
            }`}
            onClick={() => setActivePage('Explanations')}
            type="button"
          >
            <span className="nav-symbol" aria-hidden="true">
              ▤
            </span>
            Analysis results
          </button>
        </nav>

        <div className="sidebar-footer">
          <span
            className={`connection-dot ${health ? 'online' : ''}`}
          />

          <div>
            <strong>Analysis service</strong>
            <span>
              {health
                ? 'Connected'
                : apiError || 'Checking connection'}
            </span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <div>
            <p className="breadcrumb">
              Vehicle insurance / {activePage}
            </p>

            <h1>
              {activePage === 'Overview'
                ? 'Overview'
                : activePage === 'New Claim'
                  ? 'New claim analysis'
                  : 'Analysis results'}
            </h1>
          </div>

          <span
            className={`service-tag ${
              health ? 'service-ready' : ''
            }`}
          >
            <span className="connection-dot" />
            {health ? 'Service connected' : 'Service unavailable'}
          </span>
        </header>

        {activePage === 'Overview' && (
          <>
            <section className="welcome-section">
              <p className="eyebrow">VEHICLE INSURANCE</p>

              <h2>
                Understand the factors behind a claim prediction.
              </h2>

              <p className="welcome-description">
                Assess a vehicle claim using a machine-learning
                model and explore alternative input values that may
                change its predicted outcome.
              </p>

              <div className="welcome-actions">
                <button
                  className="primary-button"
                  onClick={() => setActivePage('New Claim')}
                  type="button"
                >
                  Start claim analysis
                </button>

                <button
                  className="secondary-button"
                  onClick={() => setActivePage('Explanations')}
                  type="button"
                >
                  View analysis results
                </button>
              </div>
            </section>

            <section className="overview-section">
              <div className="section-heading">
                <div>
                  <h3>How it works</h3>
                  <p>
                    From claim details to alternative model outcomes.
                  </p>
                </div>
              </div>

              <div className="process-grid">
                <article className="process-card">
                  <span className="step-number">01</span>
                  <h4>Enter claim details</h4>
                  <p>
                    Provide vehicle, policy, accident and financial
                    information for analysis.
                  </p>
                </article>

                <article className="process-card">
                  <span className="step-number">02</span>
                  <h4>Review the prediction</h4>
                  <p>
                    See the outcome predicted by the trained
                    classification model.
                  </p>
                </article>

                <article className="process-card">
                  <span className="step-number">03</span>
                  <h4>Explore alternatives</h4>
                  <p>
                    Compare hypothetical claim and repair amounts
                    associated with a different model prediction.
                  </p>
                </article>
              </div>
            </section>

            
          </>
        )}

        {activePage === 'New Claim' && (
          <section className="form-panel">
            <div className="section-heading form-heading">
              <div>
                <h2>Vehicle claim details</h2>
                <p>
                  Enter the details below to generate a prediction
                  and counterfactual explanations.
                </p>
              </div>
            </div>

            {!health && (
              <p className="notice error-notice" role="alert">
                {apiError ||
                  'The analysis service is not connected. Start the backend and try again.'}
              </p>
            )}

            <form onSubmit={handleSubmit}>
              <div className="form-section-title">
                Vehicle and driver
              </div>

              <div className="form-grid">
                <label>
                  Vehicle age (years)
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="1"
                    value={claim.vehicle_age}
                    onChange={(event) =>
                      updateClaim(
                        'vehicle_age',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>

                <label>
                  Driver age (years)
                  <input
                    type="number"
                    min="18"
                    max="120"
                    step="1"
                    value={claim.driver_age}
                    onChange={(event) =>
                      updateClaim(
                        'driver_age',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>

                <label>
                  Vehicle value (₹)
                  <input
                    type="number"
                    min="0.01"
                    step="0.01"
                    value={claim.vehicle_value}
                    onChange={(event) =>
                      updateClaim(
                        'vehicle_value',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>

                <label>
                  Previous claims
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="1"
                    value={claim.previous_claims}
                    onChange={(event) =>
                      updateClaim(
                        'previous_claims',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>
              </div>

              <div className="form-section-title">
                Claim and repair
              </div>

              <div className="form-grid">
                <label>
                  Claim amount (₹)
                  <input
                    type="number"
                    min="0.01"
                    step="0.01"
                    value={claim.claim_amount}
                    onChange={(event) =>
                      updateClaim(
                        'claim_amount',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>

                <label>
                  Repair cost (₹)
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    value={claim.repair_cost}
                    onChange={(event) =>
                      updateClaim(
                        'repair_cost',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>
              </div>

              <div className="form-section-title">
                Policy and incident
              </div>

              <div className="form-grid">
                <label>
                  Policy duration (months)
                  <input
                    type="number"
                    min="0"
                    max="1200"
                    step="1"
                    value={claim.policy_duration_months}
                    onChange={(event) =>
                      updateClaim(
                        'policy_duration_months',
                        Number(event.target.value),
                      )
                    }
                    required
                  />
                </label>

                <label>
                  Accident severity
                  <select
                    value={claim.accident_severity}
                    onChange={(event) =>
                      updateClaim(
                        'accident_severity',
                        event.target.value,
                      )
                    }
                  >
                    <option value="minor">Minor</option>
                    <option value="moderate">Moderate</option>
                    <option value="major">Major</option>
                  </select>
                </label>

                <label>
                  Documents complete?
                  <select
                    value={claim.documents_complete}
                    onChange={(event) =>
                      updateClaim(
                        'documents_complete',
                        Number(event.target.value),
                      )
                    }
                  >
                    <option value={1}>Yes</option>
                    <option value={0}>No</option>
                  </select>
                </label>

                <label>
                  Police report available?
                  <select
                    value={claim.police_report_available}
                    onChange={(event) =>
                      updateClaim(
                        'police_report_available',
                        Number(event.target.value),
                      )
                    }
                  >
                    <option value={1}>Yes</option>
                    <option value={0}>No</option>
                  </select>
                </label>

                <label>
                  Number of alternatives
                  <select
                    value={claim.total_cfs}
                    onChange={(event) =>
                      updateClaim(
                        'total_cfs',
                        Number(event.target.value),
                      )
                    }
                  >
                    <option value={1}>1 alternative</option>
                    <option value={2}>2 alternatives</option>
                    <option value={3}>3 alternatives</option>
                    <option value={4}>4 alternatives</option>
                    <option value={5}>5 alternatives</option>
                  </select>
                </label>
              </div>

              {explanationError && (
                <p className="notice error-notice" role="alert">
                  {explanationError}
                </p>
              )}

              <div className="form-actions">
                <button
                  className="primary-button"
                  type="submit"
                  disabled={loading || !health}
                >
                  {loading
                    ? 'Analysing claim...'
                    : 'Assess claim'}
                </button>

                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setClaim(initialClaim)
                    setExplanationError('')
                  }}
                >
                  Reset form
                </button>
              </div>
            </form>

            
          </section>
        )}

        {activePage === 'Explanations' && (
          <section className="results-section">
            <div className="section-heading results-heading">
              <div>
                <h2>Analysis results</h2>
                <p>
                  Review the model prediction and explore
                  alternative scenarios.
                </p>
              </div>

              <button
                className="primary-button"
                onClick={() => setActivePage('New Claim')}
                type="button"
              >
                New analysis
              </button>
            </div>

            {!explanation && (
              <div className="empty-panel">
                <h3>No results to display</h3>
                <p>
                  Start a claim analysis to view its prediction
                  and counterfactual alternatives here.
                </p>

                <button
                  className="primary-button"
                  onClick={() => setActivePage('New Claim')}
                  type="button"
                >
                  Start analysis
                </button>
              </div>
            )}

            {explanation && (
              <>
                <article
                  className={`prediction-panel prediction-${decisionType}`}
                >
                  <div className="prediction-main">
                    <div className="prediction-heading">
                      {decisionType === 'approved' ? (
                        <span
                          className="decision-icon decision-icon-approved"
                          aria-hidden="true"
                        >
                          ✓
                        </span>
                      ) : decisionType === 'rejected' ? (
                        <span
                          className="decision-icon decision-icon-rejected"
                          aria-hidden="true"
                        >
                          !
                        </span>
                      ) : (
                        <span
                          className="decision-icon decision-icon-unknown"
                          aria-hidden="true"
                        >
                          i
                        </span>
                      )}

                      <div>
                        <p className="eyebrow">
                          CLAIM ASSESSMENT
                        </p>

                        <h2>
                          {decisionType === 'approved'
                            ? 'Likely Approved'
                            : decisionType === 'rejected'
                              ? 'Likely Rejected'
                              : 'Prediction Result'}
                        </h2>
                      </div>
                    </div>

                    <p className="prediction-explanation">
                      {decisionType === 'approved'
                        ? 'The model predicts that this claim will be approved.'
                        : decisionType === 'rejected'
                          ? 'The model predicts that this claim will be rejected.'
                          : `The model returned "${explanation.original_prediction}". Please review this result.`}
                    </p>

                    <p className="prediction-caveat">
                      This is an automated model prediction, not a
                      final decision by an insurance company.
                    </p>
                  </div>

                  <span
                    className={`decision-badge decision-badge-${decisionType}`}
                  >
                    {decisionType === 'approved'
                      ? 'APPROVED'
                      : decisionType === 'rejected'
                        ? 'REJECTED'
                        : 'REVIEW RESULT'}
                  </span>
                </article>

                <div className="section-heading alternatives-heading">
                  <div>
                    <h3>Counterfactual alternatives</h3>
                    <p>
                      Hypothetical input values for which the model
                      predicts an alternative outcome.
                    </p>
                  </div>
                </div>

                {explanation.counterfactuals.length === 0 ? (
                  <div className="empty-panel no-alternatives">
                    <h3>No alternative outcome was found</h3>

                    <p>
                      The model could not find a suitable combination
                      of claim and repair amounts that changes its
                      prediction for this claim.
                    </p>

                    <p>
                      This does not mean the claim is invalid. You
                      can try another claim or review the details.
                    </p>

                    <button
                      className="secondary-button"
                      onClick={() => setActivePage('New Claim')}
                      type="button"
                    >
                      Review claim details
                    </button>
                  </div>
                ) : (
                  <div className="results-grid">
                    {explanation.counterfactuals.map(
                      (item, index) => {
                        const alternativeDecision =
                          getDecisionType(item.predicted_decision)

                        return (
                          <article
                            className="result-card"
                            key={`${item.claim_amount}-${item.repair_cost}-${index}`}
                          >
                            <p className="eyebrow">
                              ALTERNATIVE {index + 1}
                            </p>

                            <h3>{item.predicted_decision}</h3>

                            <p className="result-subtitle">
                              Predicted outcome for this scenario
                            </p>

                            <dl className="result-details">
                              <div>
                                <dt>Claim amount</dt>
                                <dd>
                                  {formatAmount(item.claim_amount)}
                                </dd>
                              </div>

                              <div>
                                <dt>Repair cost</dt>
                                <dd>
                                  {formatAmount(item.repair_cost)}
                                </dd>
                              </div>
                            </dl>

                            <p
                              className={`alternative-result alternative-result-${alternativeDecision}`}
                            >
                              {alternativeDecision === 'approved'
                                ? 'Model predicts approval'
                                : alternativeDecision === 'rejected'
                                  ? 'Model predicts rejection'
                                  : 'Model outcome'}
                            </p>

                            <button
                              className="secondary-button apply-button"
                              onClick={() => applyAlternative(item)}
                              type="button"
                            >
                              Use these values
                            </button>
                          </article>
                        )
                      },
                    )}
                  </div>
                )}

                <p className="notice">
                  {explanation.disclaimer ||
                    'Counterfactuals are hypothetical model outputs. They do not establish causation or guarantee claim approval.'}
                </p>

                <button
                  className="secondary-button return-button"
                  onClick={() => setActivePage('New Claim')}
                  type="button"
                >
                  Return to claim form
                </button>
              </>
            )}
          </section>
        )}

        <footer className="main-footer">
          <span>Claim Transparency · Vehicle claim analysis</span>
          <span>
            For demonstration and research only. Not for real
            insurance decisions.
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App
