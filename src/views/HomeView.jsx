import literatureSearchingIllustration from "../assets/literature-searching-illustration.png";
import breezeLeavesIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/breeze-leaves.svg";
import confettiBurstIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/confetti-burst.svg";
import fireworkBurstIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/firework-burst.svg";
import radiantSparkleIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/radiant-sparkle.svg";
import ribbonSwirlIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/ribbon-swirl.svg";
import shootingStarIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/shooting-star.svg";
import sparkleClusterIcon from "../../web_icon_pack/svg/medical_decor_icon_pack_svg/sparkle-cluster.svg";
import microscopeIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/01_microscope.svg";
import dnaHelixIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/02_dna_helix.svg";
import cellCultureIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/03_cell_culture.svg";
import testTubeIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/04_test_tube.svg";
import clipboardIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/05_medical_clipboard.svg";
import brainNeuronIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/06_brain_neuron.svg";
import capsuleIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/07_capsule.svg";
import labFlaskIcon from "../../web_icon_pack/svg/biomedical_icon_pack_svg/08_lab_flask.svg";

const homeDecorations = [
  { src: fireworkBurstIcon, className: "home-decor-icon--top-left" },
  { src: dnaHelixIcon, className: "home-decor-icon--left-upper" },
  { src: breezeLeavesIcon, className: "home-decor-icon--left-edge" },
  { src: microscopeIcon, className: "home-decor-icon--left-card" },
  { src: cellCultureIcon, className: "home-decor-icon--left-lower" },
  { src: shootingStarIcon, className: "home-decor-icon--left-bottom" },
  { src: brainNeuronIcon, className: "home-decor-icon--right-top" },
  { src: capsuleIcon, className: "home-decor-icon--right-upper" },
  { src: ribbonSwirlIcon, className: "home-decor-icon--right-edge" },
  { src: clipboardIcon, className: "home-decor-icon--right-mid" },
  { src: labFlaskIcon, className: "home-decor-icon--right-lower" },
  { src: testTubeIcon, className: "home-decor-icon--right-bottom" },
  { src: confettiBurstIcon, className: "home-decor-icon--hero-left" },
  { src: radiantSparkleIcon, className: "home-decor-icon--hero-right" },
  { src: sparkleClusterIcon, className: "home-decor-icon--choice-right" },
];

function HomeActionButton({ label, tone, onClick }) {
  return (
    <button
      className={`primary-button home-action-button home-action-button--${tone}`}
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      <span className="home-action-button__arrow" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </button>
  );
}

export default function HomeView({ onNavigate, t }) {
  return (
    <section id="homeView" className="app-view is-active">
      <div className="home-layout">
        <div className="home-decor-icons" aria-hidden="true">
          {homeDecorations.map((icon) => (
            <img
              key={icon.className}
              className={`home-decor-icon ${icon.className}`}
              src={icon.src}
              alt=""
            />
          ))}
        </div>
        <div className="home-hero" aria-label={t("home.heroAria")}>
          <h1 className="home-hero-title">
            <span className="home-hero-line">
              Make{" "}
              <span className="home-domain-rotator" aria-hidden="true">
                <span className="home-domain home-domain--medical">medical</span>
                <span className="home-domain home-domain--biological">biological</span>
              </span>
              {" "}literature
              <span className="sr-only">medical/biological</span>
            </span>
            <span className="home-hero-line home-hero-line--action">
              research
              <span className="home-word-rotator" aria-hidden="true">
                <span className="home-word home-word--easier">easier</span>
                <span className="home-word home-word--faster">faster</span>
                <span className="home-word home-word--better">better</span>
              </span>
              <span className="sr-only">easier, faster, better</span>
            </span>
          </h1>
        </div>

        <section className="search-flow-intro" aria-labelledby="searchFlowIntroTitle">
          <div className="search-flow-intro__copy">
            <p className="search-flow-intro__eyebrow">{t("home.searchEyebrow")}</p>
            <h2 id="searchFlowIntroTitle">{t("home.searchTitle")}</h2>
            <p>{t("home.searchBody")}</p>
            <button
              className="search-flow-intro__button"
              type="button"
              onClick={() => onNavigate("search")}
              aria-label={t("home.startSearch")}
              title={t("home.startSearch")}
            >
              <span className="search-flow-intro__button-arrow" aria-hidden="true">
                <span />
              </span>
              <span className="sr-only">{t("home.startSearch")}</span>
            </button>
          </div>

          <div className="search-flow-intro__stage" aria-label={t("home.videoAria")}>
            <div className="search-flow-intro__notes" aria-hidden="true">
              <span>{t("home.noteTopic")}</span>
              <span>{t("home.noteSelect")}</span>
              <span>{t("home.noteConfirm")}</span>
            </div>
            <div className="search-flow-intro__video-window">
              <div className="search-flow-intro__video-bar">
                <span>Workflow preview</span>
                <span className="search-flow-intro__video-dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
              <video
                className="search-flow-intro__video"
                src="/videos/search-flow-intro.mp4"
                autoPlay
                muted
                loop
                playsInline
                controls
              />
            </div>
          </div>
        </section>

        <div className="choice-grid" aria-label={t("home.choiceAria")}>
          <article className="choice-card">
            <div>
              <div className="tag-row">
                <span className="tag tag-accent">{t("home.fastPath")}</span>
                <span className="tag">{t("home.directTag")}</span>
              </div>
              <h2>{t("home.directTitle")}</h2>
              <p>{t("home.directBody")}</p>
            </div>
            <HomeActionButton
              label={t("home.addLiterature")}
              tone="purple"
              onClick={() => onNavigate("direct")}
            />
          </article>

          <figure className="choice-illustration" aria-label={t("home.illustrationAria")}>
            <img src={literatureSearchingIllustration} alt={t("home.illustrationAlt")} />
          </figure>

          <article className="choice-card choice-card--search">
            <div>
              <div className="tag-row">
                <span className="tag tag-blue">{t("home.stepFlow")}</span>
                <span className="tag">{t("home.stepFlowTag")}</span>
              </div>
              <h2>{t("home.searchFlowTitle")}</h2>
              <p>{t("home.searchFlowBody")}</p>
            </div>
            <HomeActionButton
              label={t("home.startSearch")}
              tone="cyan"
              onClick={() => onNavigate("search")}
            />
          </article>
        </div>
      </div>
    </section>
  );
}
