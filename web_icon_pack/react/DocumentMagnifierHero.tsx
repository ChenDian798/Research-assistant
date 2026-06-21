import React from "react";

type Props = {
  width?: number | string;
  height?: number | string;
  className?: string;
};

export function DocumentMagnifierHero({ width = 720, height = 360, className }: Props) {
  return (
    <svg width={width} height={height} viewBox="0 0 720 360" fill="none" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="folderGrad" x1="315" y1="44" x2="329" y2="202" gradientUnits="userSpaceOnUse">
          <stop stopColor="#54E3C9"/>
          <stop offset="1" stopColor="#20BCA7"/>
        </linearGradient>
        <linearGradient id="glassGrad" x1="414" y1="83" x2="487" y2="168" gradientUnits="userSpaceOnUse">
          <stop stopColor="#F7FCFF"/>
          <stop offset="1" stopColor="#BFD8FF"/>
        </linearGradient>
        <linearGradient id="blueGrad" x1="535" y1="218" x2="653" y2="254" gradientUnits="userSpaceOnUse">
          <stop stopColor="#3B8CFF"/>
          <stop offset="1" stopColor="#2368EA"/>
        </linearGradient>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%">
          <feDropShadow dx="0" dy="14" stdDeviation="18" floodColor="#2F6DF6" floodOpacity="0.16"/>
        </filter>
        <filter id="paperShadow" x="-20%" y="-20%" width="140%" height="150%">
          <feDropShadow dx="0" dy="10" stdDeviation="10" floodColor="#8AA3C7" floodOpacity="0.18"/>
        </filter>
      </defs>
      <ellipse cx="360" cy="292" rx="286" ry="34" fill="#EAF2FF"/>
      <path d="M497 170c10-44 39-63 64-36 24 27 12 74 54 76 33 2 47 25 34 48H421c-13-48 49-48 76-88Z" fill="#EEF5FF"/>
      <circle cx="565" cy="108" r="7" fill="#B8C9E8"/>
      <circle cx="59" cy="220" r="6" fill="#8CB7FF"/>
      <path d="M655 51l6 12 12 6-12 6-6 12-6-12-12-6 12-6 6-12Z" fill="#3B7CFF"/>
      <path d="M44 82l8 16 16 8-16 8-8 16-8-16-16-8 16-8 8-16Z" fill="#FFB53D"/>
      <path d="M689 197l5 10 10 5-10 5-5 10-5-10-10-5 10-5 5-10Z" fill="#FFD35A"/>

      <g filter="url(#softShadow)">
        <rect x="516" y="240" width="137" height="26" rx="8" fill="url(#blueGrad)"/>
        <rect x="526" y="247" width="89" height="12" rx="4" fill="#EAF2FF"/>
        <rect x="493" y="266" width="143" height="27" rx="8" fill="#FFC54D"/>
        <rect x="512" y="273" width="88" height="12" rx="4" fill="#FFF8DB"/>
      </g>
      <g>
        <path d="M601 217c-7-58 12-108 61-133 10 50-10 94-61 133Z" fill="#56D8C0"/>
        <path d="M604 217c18-58 52-86 99-87-8 50-44 82-99 87Z" fill="#6BE2CB"/>
        <path d="M602 218c-22-46-15-87 16-118 24 42 16 84-16 118Z" fill="#8BF0DB"/>
        <path d="M609 208c14-21 31-42 52-68M606 207c-4-30-1-58 11-87M610 213c27-22 54-42 91-58" stroke="#25BCA8" strokeWidth="4" strokeLinecap="round" opacity=".45"/>
      </g>

      <g filter="url(#paperShadow)">
        <path d="M239 62h171c12 0 22 10 22 22v144c0 12-10 22-22 22H239c-12 0-22-10-22-22V84c0-12 10-22 22-22Z" fill="url(#folderGrad)"/>
        <path d="M241 62h78c8 0 14 4 18 10l8 13h86v36H217V86c0-13 11-24 24-24Z" fill="#62E8CF"/>
        <path d="M391 88h28" stroke="#DFFFF7" strokeWidth="5" strokeLinecap="round" opacity=".9"/>
      </g>

      <g filter="url(#paperShadow)">
        <path d="M106 91c70 11 132 0 196-18 22 64 29 142 26 235-75 17-144 15-218 1 10-68 12-141-4-218Z" fill="#F5FAFF" stroke="#A5B8DB" strokeWidth="4"/>
        <path d="M95 112c69 12 133 1 197-17 20 68 26 138 23 222-73 19-148 18-222 1 12-65 15-134 2-206Z" fill="#FFFFFF" stroke="#A5B8DB" strokeWidth="4"/>
        <path d="M79 130c70 15 137 6 202-14 23 70 31 137 28 210-77 22-154 22-232 2 14-62 18-129 2-198Z" fill="#FFFFFF" stroke="#A5B8DB" strokeWidth="4"/>
        <rect x="139" y="159" width="103" height="78" rx="9" fill="#EEF5FF" stroke="#9DB7E9" strokeWidth="5"/>
        <path d="M154 219l26-30 24 23 14-16 17 23H154Z" fill="#3378F6"/>
        <circle cx="214" cy="181" r="8" fill="#9DB7E9"/>
        <path d="M122 258h155M116 280h142M112 302h111" stroke="#9DB0D6" strokeWidth="7" strokeLinecap="round"/>
        <path d="M259 161h81M259 186h70M259 211h58" stroke="#9DB0D6" strokeWidth="7" strokeLinecap="round"/>
      </g>

      <g filter="url(#softShadow)">
        <circle cx="429" cy="151" r="59" fill="url(#glassGrad)" fillOpacity=".78" stroke="#243D85" strokeWidth="13"/>
        <path d="M472 193l86 86" stroke="#243D85" strokeWidth="18" strokeLinecap="round"/>
        <rect x="539" y="250" width="35" height="65" rx="17.5" transform="rotate(-45 539 250)" fill="#243D85"/>
        <path d="M390 126c11-19 30-29 52-27" stroke="white" strokeWidth="8" strokeLinecap="round" opacity=".82"/>
        <path d="M377 158c1-9 3-17 7-24" stroke="white" strokeWidth="8" strokeLinecap="round" opacity=".65"/>
      </g>
    </svg>
  );
}
