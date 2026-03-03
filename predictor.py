import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
import google.generativeai as genai
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictPredictor:
    """
    Analyzes geopolitical event data to predict upcoming conflicts.
    """
    
    # GDELT Event Codes for "Material Conflict" (CAMEO codes 18-20)
    CONFLICT_CODES = ['18', '19', '20'] 
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini AI integration initialized.")

    def analyze_risk(self, df):
        """
        Processes GDELT DataFrame to identify high-risk areas.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Group by geolocation to find hotspots
        df['RiskScore'] = (df['Goldstein'] * -1) + (df['Tone'] * -1) + (df['Mentions'] * 0.1)
        
        # Focus on negative sentiment/conflict events
        hotspots = df.groupby(['Lat', 'Long', 'Actor1', 'Actor2']).agg({
            'RiskScore': 'mean',
            'Tone': 'mean',
            'EventCode': 'count',
            'Source': 'first'
        }).reset_index()

        hotspots = hotspots.rename(columns={'EventCode': 'EventCount'})
        hotspots = hotspots.sort_values(by='RiskScore', ascending=False)
        
        return hotspots

    def generate_predictions(self, hotspots_df):
        """
        Generates 10 specific predicted events based on hotspots.
        """
        if hotspots_df is None or hotspots_df.empty:
            return []

        predictions = []
        top_hotspots = hotspots_df.head(10)

        for _, row in top_hotspots.iterrows():
            if self.api_key:
                # Use Gemini for enhanced reasoning
                reason, weapons, risk_lvl = self._get_gemini_insight(row)
                social_sentiment = self.get_social_sentiment(f"{row['Actor1']} and {row['Actor2']}")
            else:
                reason = self._generate_reasoning(row)
                weapons = self._predict_weapons(row['Actor1'], row['Actor2'])
                risk_lvl = self._map_risk_level(row['RiskScore'])
                social_sentiment = "X-FEED OFFLINE"
                
            pred = {
                'location': {'lat': row['Lat'], 'lon': row['Long']},
                'actors': f"{row['Actor1']} and {row['Actor2']}",
                'risk_level': risk_lvl,
                'reason': reason,
                'weapons': weapons,
                'social_sentiment': social_sentiment,
                'source': row['Source']
            }
            predictions.append(pred)

        return predictions

    # Offline intelligence database for when Gemini is unavailable
    OFFLINE_INTEL = {
        "India": {
            "history": "• **1947 Indo-Pakistani War**: Partition of British India triggered the first armed conflict over Kashmir, establishing a contested Line of Control that persists today.\n• **1962 Sino-Indian War**: China launched a surprise offensive across the Himalayan border, exposing critical defense gaps. India suffered a decisive defeat in the NEFA and Aksai Chin sectors.\n• **1999 Kargil War**: Pakistani forces infiltrated positions along the LoC in Kargil. India launched Operation Vijay, recapturing strategic peaks through intense mountain warfare at 18,000ft altitude.",
            "future": "• **Kashmir Flashpoint**: Ongoing tensions along the LoC with Pakistan remain a constant trigger for escalation, especially with increasing cross-border militant activity.\n• **LAC Border Standoff**: China continues infrastructure buildup near the Line of Actual Control. Doklam and Galwan-type incidents could recur with strategic consequences.\n• **Indian Ocean Dominance**: Rising naval competition with China's String of Pearls strategy threatens India's maritime dominance in the Indian Ocean Region."
        },
        "Ukraine": {
            "history": "• **2014 Crimea Annexation**: Russia annexed Crimea following a disputed referendum, fundamentally altering European security architecture.\n• **2014-2022 Donbas War**: Pro-Russian separatists seized control of Donetsk and Luhansk oblasts, creating a frozen conflict with over 14,000 casualties.\n• **2022 Full-Scale Invasion**: Russia launched a multi-axis invasion targeting Kyiv, Kharkiv, and southern Ukraine, triggering the largest European conflict since WWII.",
            "future": "• **Attritional Warfare**: The conflict has evolved into a grinding war of attrition with neither side capable of decisive breakthrough operations.\n• **Nuclear Escalation Risk**: Russian nuclear rhetoric remains a persistent concern as battlefield losses mount.\n• **NATO Expansion**: Finland and Sweden's NATO accession has fundamentally reshaped Northern European security dynamics."
        },
        "Russia": {
            "history": "• **Soviet-Afghan War (1979-1989)**: A decade-long counterinsurgency campaign that drained Soviet resources and contributed to the USSR's collapse.\n• **Chechen Wars (1994-2009)**: Two brutal campaigns to suppress Chechen separatism, involving significant urban warfare in Grozny.\n• **2022 Ukraine Invasion**: Full-scale military operation against Ukraine, resulting in severe Western sanctions and international isolation.",
            "future": "• **Economic Strain**: Western sanctions continue to degrade Russia's military-industrial capacity and economic stability.\n• **Arctic Militarization**: Russia is expanding military presence in the Arctic as climate change opens new strategic waterways.\n• **Central Asian Influence**: Growing competition with China and Turkey for influence in former Soviet republics."
        },
        "Israel": {
            "history": "• **1948 War of Independence**: Israel declared statehood, immediately facing invasion from five Arab states. Established borders through armistice agreements.\n• **1967 Six-Day War**: Preemptive strikes against Egypt, Syria, and Jordan resulted in Israeli control of the Sinai, Gaza, West Bank, and Golan Heights.\n• **2023 Gaza War**: Hamas launched unprecedented attack on October 7th, triggering the most intensive Israeli military operation in Gaza since 1967.",
            "future": "• **Iran Nuclear Threat**: Iran's advancing nuclear program remains Israel's primary strategic concern, with potential for preemptive military action.\n• **Hezbollah Northern Front**: Lebanese Hezbollah's 150,000+ rocket arsenal poses a significant multi-front warfare risk.\n• **Regional Normalization**: Abraham Accords expansion could reshape Middle Eastern alliances but faces setbacks from Gaza conflict."
        },
        "Iran": {
            "history": "• **Iran-Iraq War (1980-1988)**: One of the 20th century's longest conventional wars, resulting in over 1 million casualties and extensive use of chemical weapons.\n• **1979 Islamic Revolution**: Overthrow of the Shah established a theocratic republic, fundamentally altering Middle Eastern geopolitics.\n• **Proxy Warfare Network**: Iran built an extensive network of proxy forces across the Middle East including Hezbollah, Hamas, and Houthi groups.",
            "future": "• **Nuclear Threshold**: Iran continues uranium enrichment approaching weapons-grade levels, escalating tensions with Israel and the West.\n• **Proxy Network Disruption**: Israeli operations have degraded Iranian proxy capabilities, potentially forcing direct confrontation.\n• **Internal Instability**: Ongoing protest movements challenge regime legitimacy and could trigger unpredictable political shifts."
        },
        "China": {
            "history": "• **Korean War (1950-1953)**: China's intervention with 300,000+ troops pushed UN forces back from the Yalu River, establishing the current Korean DMZ.\n• **1962 Sino-Indian War**: Brief but decisive border conflict that established Chinese control over Aksai Chin.\n• **1979 Sino-Vietnamese War**: China invaded Vietnam in response to Vietnam's invasion of Cambodia, suffering unexpectedly high casualties.",
            "future": "• **Taiwan Contingency**: Military preparations for potential Taiwan reunification by force remain China's primary strategic focus with 2027 as a key readiness date.\n• **South China Sea Expansion**: Ongoing island-building and militarization programs threaten freedom of navigation and regional stability.\n• **US Strategic Competition**: Intensifying great power rivalry spanning military, economic, and technological domains."
        },
        "Taiwan": {
            "history": "• **1949 Republic of China Retreat**: Nationalist forces retreated to Taiwan after losing the Chinese Civil War, establishing a separate government.\n• **1954-55 & 1958 Taiwan Strait Crises**: China shelled Taiwanese-held islands, bringing the US and China to the brink of nuclear confrontation.\n• **1996 Taiwan Strait Crisis**: China conducted missile tests near Taiwan during presidential elections; US deployed two carrier battle groups in response.",
            "future": "• **Invasion Scenario**: Chinese military modernization and amphibious capabilities are being developed with a Taiwan contingency as the primary design requirement.\n• **Semiconductor Leverage**: Taiwan's dominance in advanced chip manufacturing (TSMC) makes it a critical node in global technology supply chains.\n• **US Security Commitment**: Ambiguity over US defense obligations creates strategic uncertainty that could invite miscalculation."
        },
        "Pakistan": {
            "history": "• **1971 Bangladesh Liberation War**: Pakistan's military operation in East Pakistan triggered Indian intervention, resulting in the creation of Bangladesh.\n• **1998 Nuclear Tests**: Pakistan conducted nuclear weapons tests in response to India, establishing mutual nuclear deterrence on the subcontinent.\n• **War on Terror (2001-present)**: Pakistan served as both ally and adversary in the US-led campaign, conducting major operations in its tribal areas.",
            "future": "• **Kashmir Tensions**: Persistent territorial dispute with India over Kashmir remains a nuclear flashpoint.\n• **Afghanistan Spillover**: Taliban governance in Afghanistan creates cross-border security challenges and refugee pressures.\n• **Economic Crisis**: Severe economic instability and IMF dependency could trigger social unrest and political instability."
        },
        "North Korea": {
            "history": "• **Korean War (1950-1953)**: North Korean invasion of South Korea triggered a devastating three-year conflict with 2.5 million casualties.\n• **Nuclear Program**: Conducted six nuclear tests (2006-2017), developing thermonuclear warheads and ICBM delivery systems.\n• **Provocations Cycle**: Decades of periodic military provocations including the sinking of ROKS Cheonan (2010) and shelling of Yeonpyeong Island.",
            "future": "• **ICBM Capability**: Continued development of solid-fuel ICBMs capable of striking the US mainland increases deterrence but also escalation risks.\n• **Regime Stability**: Succession concerns and economic isolation create unpredictable internal dynamics.\n• **Russia-DPRK Alliance**: Deepening military cooperation with Russia, including alleged ammunition supplies for the Ukraine conflict."
        },
        "South Korea": {
            "history": "• **Korean War (1950-1953)**: Devastating conflict that killed millions and divided the Korean Peninsula along the 38th parallel.\n• **Cold War Frontline**: Decades of military buildup along the DMZ with periodic North Korean provocations and assassination attempts.\n• **Vietnam War Participation**: South Korea deployed over 300,000 troops to Vietnam, the second-largest foreign contingent after the US.",
            "future": "• **North Korean Threat**: Pyongyang's advancing nuclear and missile capabilities require constant military readiness and allied coordination.\n• **US Alliance Evolution**: Changing regional dynamics require adaptation of the US-ROK alliance framework.\n• **Japan Relations**: Historical tensions complicate trilateral security cooperation despite shared threats."
        },
        "Turkey": {
            "history": "• **WWI & Gallipoli**: Ottoman Empire's defense of the Dardanelles against Allied invasion became a foundational national myth.\n• **Cyprus Intervention (1974)**: Turkish military invasion of Cyprus in response to a Greek-backed coup, resulting in the island's partition.\n• **Kurdish Conflict (1984-present)**: Prolonged counterinsurgency campaign against PKK militants in southeastern Turkey and northern Iraq/Syria.",
            "future": "• **Syria Buffer Zone**: Turkish military presence in northern Syria aims to prevent Kurdish autonomous zones but creates friction with Russia and the US.\n• **NATO Tensions**: Turkey's S-400 acquisition and independent foreign policy strain alliance cohesion.\n• **Eastern Mediterranean**: Competing maritime claims with Greece and Cyprus risk naval confrontation over energy resources."
        },
        "Syria": {
            "history": "• **Syrian Civil War (2011-present)**: Arab Spring protests escalated into a devastating multi-party civil war with over 500,000 casualties and 13 million displaced.\n• **ISIS Caliphate (2014-2019)**: Islamic State seized large swaths of eastern Syria, requiring international military intervention to defeat.\n• **Russian Intervention (2015)**: Russia's military intervention decisively shifted the conflict in favor of the Assad regime.",
            "future": "• **Frozen Conflict**: Syria remains fragmented between regime, Kurdish, Turkish, and rebel-controlled zones with no political resolution in sight.\n• **Iranian Entrenchment**: Iran's military presence in Syria threatens Israeli security and risks broader regional escalation.\n• **Reconstruction Crisis**: Estimated $400 billion reconstruction cost with no international consensus on funding or political framework."
        },
    }

    def get_regional_intelligence(self, region_name):
        """
        Generates a detailed intelligence report for a specific region.
        Uses Gemini AI when available, falls back to offline database.
        """
        # Try Gemini first
        if self.api_key:
            try:
                prompt = f"""
                ACT AS A SENIOR GEOPOLITICAL INTELLIGENCE ANALYST.
                PROVIDE A BRIEF, TACTICAL INTELLIGENCE REPORT FOR: {region_name}

                FORMAT:
                1. HISTORY: A summary of major historical conflicts in 3 bullet points. Use bullet format (• bold title: description).
                2. FUTURE: A predictive analysis of future risks in 3 bullet points. Use bullet format (• bold title: description).

                BE CONCISE. USE TACTICAL LANGUAGE.
                """
                response = self.model.generate_content(prompt)
                data = response.text.split("2. FUTURE:")
                history = data[0].replace("1. HISTORY:", "").strip()
                future = data[1].strip() if len(data) > 1 else "Analysis pending..."
                
                return {"history": history, "future": future}
            except Exception as e:
                logger.warning(f"Gemini unavailable for {region_name}: {e}. Using offline intel.")

        # Fallback to offline database
        if region_name in self.OFFLINE_INTEL:
            return self.OFFLINE_INTEL[region_name]
        
        # Generic fallback for unlisted countries
        return {
            "history": f"• **Historical Context**: {region_name} has experienced various periods of conflict and political transformation throughout its history.\n• **Colonial Legacy**: Many of the region's current borders and tensions trace back to colonial-era decisions and post-independence power struggles.\n• **Modern Challenges**: The region faces ongoing security challenges related to territorial disputes, ethnic tensions, or resource competition.",
            "future": f"• **Regional Stability**: {region_name}'s strategic position makes it vulnerable to shifts in great power competition and regional alliance dynamics.\n• **Economic Pressures**: Global economic volatility and climate change create compounding risks for internal stability.\n• **Security Environment**: Evolving threat landscape requires adaptive defense posture and multilateral engagement."
        }

    def get_social_sentiment(self, actors):
        """
        Simulates X (Twitter) sentiment analysis using Gemini based on recent news.
        """
        if not self.api_key:
            return "SIGNAL OFFLINE"

        prompt = f"""
        ACT AS A DIGITAL SENTIMENT ANALYST.
        ANALYZE THE CURRENT 'DIGITAL MOOD' ON X (TWITTER) FOR THESE ACTORS: {actors}
        
        CONSIDER:
        - Recent hashtags, viral propaganda, and public outcry.
        - The 'vibes' of the digital conversation.
        
        FORMAT: A single, punchy, tactical sentence describing the social sentiment.
        EG: 'Digital sphere flooded with #StopWar tags; Russian botnets active in neutralization campaigns.'
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return "SOCIAL FEED NOISY - ANALYSIS INCOMPLETE"

    def generate_humorous_headline(self, hotspots_summary):
        """
        Generates a humorous but 'real' news headline based on current hotspots.
        """
        if not self.api_key:
            return "BREAKING: System failing to find humor in global catastrophe."

        prompt = f"""
        ACT AS A SATIRICAL BUT RELEVANT NEWS ANCHOR (think Jon Stewart meets a War Room General).
        GENERATE A PUNCHY, HUMOROUS, BUT REAL HEADLINE BASED ON THESE CURRENT HOTSPOTS: {hotspots_summary}

        RULES:
        1. Must be based on the ACTORS provided.
        2. Must be funny/satirical but grounded in the reality of the conflict.
        3. One sentence only.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return "BREAKING: AI refusing to joke about the current state of the world."

    def _get_gemini_insight(self, row):
        """Uses Gemini to synthesize event data into a specific prediction."""
        prompt = f"""
        Analyze this geopolitical event data from GDELT:
        Actors: {row['Actor1']} and {row['Actor2']}
        Tone Score: {row['Tone']} (Lower is more aggressive)
        Risk Score: {row['RiskScore']}
        Event Count: {row['EventCount']}

        Based on this, provide:
        1. A detailed (2-3 sentence) tactical prediction of the next likely event/escalation.
        2. A comma-separated list of 5 likely weapon systems or strategies involved.
        3. The 'Tactical Risk Level' (CRITICAL, HIGH, MODERATE).

        Format: Prediction | Weapons | RiskLevel
        """
        try:
            response = self.model.generate_content(prompt)
            parts = response.text.split('|')
            if len(parts) >= 3:
                return parts[0].strip(), parts[1].strip(), parts[2].strip()
            return response.text.strip(), "Specialized Military Assets", "MODERATE"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_reasoning(row), self._predict_weapons(row['Actor1'], row['Actor2']), self._map_risk_level(row['RiskScore'])

    def _generate_reasoning(self, row):
        tone = "highly aggressive" if row['Tone'] < -5 else "escalating"
        return f"Recent event frequency ({row['EventCount']}) between {row['Actor1']} and {row['Actor2']} shows {tone} diplomatic posture with significant negative sentiment."

    def _map_risk_level(self, score):
        if score > 15: return "CRITICAL"
        if score > 10: return "HIGH"
        return "MODERATE"

    def _predict_weapons(self, actor1, actor2):
        # Heuristic based on actor types
        world_powers = ['USA', 'RUS', 'CHN', 'ISR', 'IRN', 'UKR']
        if any(p in str(actor1) or p in str(actor2) for p in world_powers):
            return "Precision Guided Munitions, UAVs, Electronic Warfare"
        return "Small Arms, IEDs, Light Artillery"

if __name__ == "__main__":
    # Mock testing logic
    predictor = ConflictPredictor()
    mock_data = pd.DataFrame({
        'Lat': [50.45, 31.76],
        'Long': [30.52, 35.21],
        'Actor1': ['UKR', 'ISR'],
        'Actor2': ['RUS', 'IRN'],
        'Goldstein': [-10, -8],
        'Tone': [-12, -7],
        'Mentions': [100, 50],
        'Source': ['http://news.com', 'http://news.com']
    })
    
    hotspots = predictor.analyze_risk(mock_data)
    preds = predictor.generate_predictions(hotspots)
    for p in preds:
        print(f"PREDICTION: {p['actors']} in {p['location']} - {p['risk_level']}")
