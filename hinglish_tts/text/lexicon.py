"""Word lists that take priority over the rule-based transliterator.

* HINDI: the most frequent Hindi words with the spelling most Hinglish writers
  use. The first spelling is canonical (used for training text and inference);
  the others are common alternatives, used as spelling augmentation so the model
  also reads "nhi"-style inputs sensibly.
* ENGLISH_LOANWORDS: English words that Hindi corpora write in Devanagari. Mapping
  them back to English spelling turns monolingual Hindi transcripts into real
  code-mixed text ("mera computer kharab hai"), which is what users type.
* LETTER_NAMES: Devanagari spellings of English letter names, to turn acronyms
  written out in Devanagari ("पीएम") back into "PM".
* An optional large lexicon (Aksharantar, AI4Bharat) of attested romanizations,
  built on the training machine by `build_aksharantar_lexicon`.
"""

from __future__ import annotations

import collections
import io
import json
import logging
import unicodedata
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _entries(table: str) -> dict[str, tuple[str, ...]]:
    """Parse "देवनागरी canonical variant1 variant2" lines."""
    out: dict[str, tuple[str, ...]] = {}
    for line in table.strip().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        native, *roman = line.split()
        out[unicodedata.normalize("NFC", native)] = tuple(roman)
    return out


HINDI = _entries(
    """
    # pronouns
    मैं main
    मैंने maine
    मुझे mujhe mujhey
    मुझको mujhko
    मुझसे mujhse
    मेरा mera
    मेरी meri
    मेरे mere
    हम hum ham
    हमें humein hume hamein
    हमने humne hamne
    हमसे humse
    हमारा hamara hamaara
    हमारी hamari hamaari
    हमारे hamare hamaare
    तुम tum
    तुम्हें tumhe tumhein
    तुमने tumne
    तुमसे tumse
    तुम्हारा tumhara
    तुम्हारी tumhari
    तुम्हारे tumhare
    तू tu
    तुझे tujhe
    तेरा tera
    तेरी teri
    तेरे tere
    आप aap ap
    आपका aapka apka
    आपकी aapki apki
    आपके aapke apke
    आपको aapko apko
    आपने aapne
    आपसे aapse
    वह woh wo vah
    वो woh wo
    वे ve
    यह yeh ye yah
    ये ye yeh
    उसका uska
    उसकी uski
    उसके uske
    उसे use usey
    उसको usko
    उसने usne
    उससे usse
    इसका iska
    इसकी iski
    इसके iske
    इसे ise isey
    इसको isko
    इसने isne
    इससे isse
    उनका unka
    उनकी unki
    उनके unke
    उन्हें unhe unhein
    उन्होंने unhone unhonne
    इनका inka
    इन्हें inhe
    इन्होंने inhone
    अपना apna
    अपनी apni
    अपने apne
    खुद khud
    # question words
    कौन kaun kon
    क्या kya kia
    क्यों kyun kyon kyu
    क्यूँ kyun kyu
    कैसे kaise kese
    कैसा kaisa
    कैसी kaisi
    कब kab
    कहाँ kahan kahaan
    कहां kahan kahaan
    किधर kidhar
    कितना kitna
    कितनी kitni
    कितने kitne
    किसका kiska
    किसको kisko
    किसी kisi
    कोई koi
    कुछ kuch kuchh
    सब sab
    सभी sabhi
    # auxiliaries
    है hai
    हैं hain hai
    था tha
    थी thi
    थे the
    हूँ hoon hun hu
    हूं hoon hun hu
    हो ho
    होगा hoga
    होगी hogi
    होंगे honge
    होता hota
    होती hoti
    होते hote
    होना hona
    हुआ hua
    हुई hui
    हुए hue huye
    रहा raha
    रही rahi
    रहे rahe
    रहना rehna rahna
    सकता sakta
    सकती sakti
    सकते sakte
    चाहिए chahiye chaahiye
    चाहिये chahiye chaahiye
    चाहता chahta
    चाहती chahti
    # particles and connectives
    नहीं nahi nahin nhi
    नही nahi nahin
    ना na
    न na
    मत mat
    और aur or
    या ya
    लेकिन lekin
    पर par
    मगर magar
    तो toh to
    भी bhi
    ही hi
    सिर्फ sirf
    सिर्फ़ sirf
    बस bas
    फिर phir fir
    अब ab
    तब tab
    जब jab
    अगर agar
    कि ki
    की ki
    का ka
    के ke
    को ko
    से se
    ने ne
    में mein me
    पे pe
    तक tak
    लिए liye
    लिये liye
    साथ saath sath
    बाद baad bad
    पहले pehle pahle
    अभी abhi
    कभी kabhi
    यहाँ yahan yahaan
    यहां yahan yahaan
    वहाँ wahan vahan
    वहां wahan vahan
    इधर idhar
    उधर udhar
    ऊपर upar oopar
    नीचे neeche niche
    अंदर andar
    बाहर bahar baahar
    क्योंकि kyunki kyonki
    इसलिए isliye isliey
    इसलिये isliye
    जैसे jaise
    जैसा jaisa
    वैसे waise vaise
    वैसा waisa vaisa
    ऐसे aise
    ऐसा aisa
    ऐसी aisi
    शायद shayad
    बिल्कुल bilkul
    एकदम ekdum
    ज़रूर zaroor jaroor
    जरूर zaroor jaroor
    मतलब matlab
    यानी yaani yani
    # adjectives and adverbs
    बहुत bahut bohot bahot
    ज़्यादा zyada jyada zyaada
    ज्यादा zyada jyada
    कम kam
    थोड़ा thoda
    थोड़ी thodi
    अच्छा accha acha achha
    अच्छी acchi achhi achi
    अच्छे acche achhe ache
    बुरा bura
    ठीक theek thik
    सही sahi
    ग़लत galat
    गलत galat
    बड़ा bada
    बड़ी badi
    बड़े bade
    छोटा chhota chota
    छोटी chhoti choti
    छोटे chhote chote
    नया naya
    नई nayi nai
    नए naye
    पुराना purana
    जल्दी jaldi
    धीरे dheere dhire
    सबसे sabse
    दोनों dono donon
    # greetings
    जी ji
    हाँ haan han
    हां haan han
    नमस्ते namaste
    नमस्कार namaskar
    धन्यवाद dhanyavaad dhanyawad
    शुक्रिया shukriya
    # verbs
    करना karna
    करता karta
    करती karti
    करते karte
    करो karo
    करें karein karen
    किया kiya
    कर kar
    करके karke
    जाना jaana jana
    जाता jaata jata
    जाती jaati jati
    जाते jaate jate
    जाओ jao
    गया gaya
    गई gayi gai
    गए gaye
    गये gaye
    आना aana ana
    आता aata ata
    आती aati ati
    आते aate ate
    आओ aao
    आया aaya aya
    आई aayi aai
    आए aaye
    आये aaye
    देना dena
    दिया diya
    लेना lena
    लिया liya
    बोलना bolna
    बोला bola
    बोलो bolo
    कहा kaha
    कहना kehna kahna
    बताओ batao
    बताना batana
    सुनो suno
    देखो dekho
    देखना dekhna
    समझ samajh
    चलो chalo
    चल chal
    मिलना milna
    मिलते milte
    # people and things
    दोस्त dost
    दोस्तों doston
    यार yaar yar
    भाई bhai
    बहन behen bahan
    माँ maa
    मां maa
    पापा papa
    बच्चे bacche bachche
    लोग log
    लोगों logon
    आदमी aadmi
    घर ghar
    काम kaam
    पैसे paise
    पैसा paisa
    रुपये rupaye
    रुपए rupaye
    खाना khana khaana
    पानी paani pani
    चाय chai
    प्यार pyaar pyar
    दिल dil
    ज़िंदगी zindagi jindagi
    जिंदगी zindagi jindagi
    दुनिया duniya
    हिंदी hindi
    भारत bharat bhaarat
    मौसम mausam
    # time
    आज aaj
    कल kal
    परसों parson
    दिन din
    रात raat
    सुबह subah
    शाम shaam sham
    समय samay
    वक़्त waqt vakt
    वक्त waqt vakt
    घंटे ghante
    # numbers that appear as words in transcripts
    एक ek
    दो do
    तीन teen
    चार chaar char
    पाँच paanch panch
    पांच paanch panch
    छह chhah chhe
    सात saat
    आठ aath
    नौ nau
    दस das
    सौ sau
    हज़ार hazaar hazar
    हजार hazaar hazar
    लाख lakh
    करोड़ crore karod
    """
)

ENGLISH_LOANWORDS = _entries(
    """
    फ़ोन phone
    फोन phone
    मोबाइल mobile
    कंप्यूटर computer
    कम्प्यूटर computer
    इंटरनेट internet
    ऑनलाइन online
    ऑफिस office
    ऑफ़िस office
    स्कूल school
    कॉलेज college
    यूनिवर्सिटी university
    मीटिंग meeting
    टाइम time
    प्रॉब्लम problem
    प्रोब्लम problem
    सर sir
    मैडम madam
    डॉक्टर doctor
    टीचर teacher
    ट्रेन train
    टिकट ticket
    होटल hotel
    पार्टी party
    फ़िल्म film
    फिल्म film
    मूवी movie
    वीडियो video
    मैसेज message
    ईमेल email
    डेटा data
    सिस्टम system
    प्रोजेक्ट project
    कंपनी company
    मैनेजर manager
    रिपोर्ट report
    बैंक bank
    कार्ड card
    पुलिस police
    स्टेशन station
    टेक्नोलॉजी technology
    सॉफ्टवेयर software
    ऐप app
    वेबसाइट website
    लैपटॉप laptop
    प्लीज़ please
    प्लीज please
    सॉरी sorry
    थैंक्स thanks
    थैंक्यू thankyou
    ओके okay
    हेलो hello
    हैलो hello
    बाय bye
    शेयर share
    मार्केट market
    बिज़नेस business
    बिजनेस business
    सर्विस service
    प्रोवाइडर provider
    गाइड guide
    ट्रैवल travel
    ट्रैवल्स travels
    फीचर feature
    मशीन machine
    इंजीनियर engineer
    टेक्नोलॉजी technology
    ट्रैक्टर tractor
    हार्वेस्टर harvester
    प्राइवेट private
    लिमिटेड limited
    सर्टिफिकेट certificate
    लाइसेंस license
    ड्राइविंग driving
    ड्राइवर driver
    फोटो photo
    फ़ोटो photo
    कॉपी copy
    फॉर्म form
    फ़ॉर्म form
    नंबर number
    लाइन line
    फ्री free
    मैच match
    टीम team
    गेम game
    स्टार्ट start
    सेंटर center
    बिल bill
    रिचार्ज recharge
    नेटवर्क network
    सिम sim
    चार्जर charger
    बैटरी battery
    कैमरा camera
    हॉस्पिटल hospital
    मेडिकल medical
    क्लास class
    टेस्ट test
    रिजल्ट result
    पेपर paper
    एग्जाम exam
    होमवर्क homework
    ट्यूशन tuition
    कोचिंग coaching
    स्टूडेंट student
    स्टूडेंट्स students
    कार car
    बाइक bike
    स्कूटर scooter
    रोड road
    ट्रैफिक traffic
    सिग्नल signal
    पार्किंग parking
    शॉपिंग shopping
    मॉल mall
    ऑफर offer
    डिस्काउंट discount
    प्राइस price
    ऑर्डर order
    डिलीवरी delivery
    पार्सल parcel
    कस्टमर customer
    सपोर्ट support
    अकाउंट account
    पासवर्ड password
    लॉगिन login
    अपडेट update
    डाउनलोड download
    अपलोड upload
    लिंक link
    चैनल channel
    न्यूज़ news
    न्यूज news
    प्रोग्राम program
    प्रोग्रामिंग programming
    कोड code
    फाइल file
    फ़ाइल file
    फोल्डर folder
    विंडो window
    स्क्रीन screen
    बटन button
    क्लिक click
    टाइप type
    ट्यूटोरियल tutorial
    कमांड command
    टर्मिनल terminal
    सर्वर server
    डेटाबेस database
    वर्जन version
    सेटिंग्स settings
    मेनू menu
    ऑप्शन option
    बैग bag
    सिलेंडर cylinder
    गैस gas
    पेट्रोल petrol
    डीज़ल diesel
    डीजल diesel
    किलोमीटर kilometer
    प्लान plan
    प्लानिंग planning
    आइडिया idea
    टॉपिक topic
    सेक्शन section
    डाउन down
    स्लो slow
    फास्ट fast
    हैंग hang
    ओपन open
    क्लोज close
    चेक check
    सेंड send
    कॉल call
    बुक book
    बुकिंग booking
    कैंसल cancel
    """
)

# English letter names as Indians write them in Devanagari.
LETTER_NAMES: dict[str, str] = {
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "एफ़": "F",
    "जी": "G", "एच": "H", "आई": "I", "जे": "J", "के": "K", "एल": "L", "एम": "M",
    "एन": "N", "ओ": "O", "पी": "P", "क्यू": "Q", "आर": "R", "एस": "S", "टी": "T",
    "यू": "U", "वी": "V", "डब्ल्यू": "W", "डबल्यू": "W", "एक्स": "X", "वाई": "Y",
    "ज़ेड": "Z", "जेड": "Z",
}  # fmt: skip
LETTER_NAMES = {unicodedata.normalize("NFC", k): v for k, v in LETTER_NAMES.items()}
# Real words that happen to spell out letter names.
ACRONYM_BLOCKLIST = {unicodedata.normalize("NFC", w) for w in ("बीबी", "जीजी", "सीसी", "पीपी")}


def acronym_of(word: str) -> str | None:
    """"पीएम" -> "PM" when the whole word segments into >= 2 English letter names."""
    word = unicodedata.normalize("NFC", word)
    if word in HINDI or word in ACRONYM_BLOCKLIST:
        return None
    names = sorted(LETTER_NAMES, key=len, reverse=True)
    letters: list[str] = []
    i = 0
    while i < len(word):
        for name in names:
            if word.startswith(name, i):
                letters.append(LETTER_NAMES[name])
                i += len(name)
                break
        else:
            return None
    return "".join(letters) if len(letters) >= 2 else None


def load_lexicon_tsv(path: str | Path) -> dict[str, str]:
    """Read a "native<TAB>roman[<TAB>count]" lexicon."""
    lexicon: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                lexicon[unicodedata.normalize("NFC", parts[0])] = parts[1]
    return lexicon


# Aksharantar rows come from several sources; human-annotated ones win ties.
_SOURCE_WEIGHT = {"Dakshina": 3.0, "AK-Freq": 3.0, "AK-Uni": 2.0, "AK-NEF": 1.5, "AK-NEI": 1.5}


def build_aksharantar_lexicon(
    out_path: str | Path,
    repo: str = "ai4bharat/Aksharantar",
    filename: str = "hin.zip",
    min_votes: float = 2.0,
) -> Path:
    """Majority-vote the attested romanization of every Hindi word in Aksharantar.

    Writes "native<TAB>roman<TAB>votes" lines. Runs on the training machine; the
    zip is ~40 MB and never needs to exist locally.
    """
    from huggingface_hub import hf_hub_download

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    zpath = hf_hub_download(repo, filename, repo_type="dataset")
    votes: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if not member.endswith((".json", ".jsonl")):
                continue
            with zf.open(member) as raw:
                for line in io.TextIOWrapper(raw, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    native = unicodedata.normalize("NFC", str(row.get("native word", "")).strip())
                    roman = str(row.get("english word", "")).strip().lower()
                    if not native or not roman.isascii() or not roman.isalpha():
                        continue
                    votes[native][roman] += _SOURCE_WEIGHT.get(str(row.get("source", "")), 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for native, counter in votes.items():
            # Ties go to the shorter spelling, the more casual one.
            roman, n = max(counter.items(), key=lambda kv: (kv[1], -len(kv[0])))
            if n >= min_votes:
                f.write(f"{native}\t{roman}\t{n:g}\n")
                kept += 1
    logger.info(f"aksharantar lexicon: {kept} of {len(votes)} words -> {out_path}")
    return out_path
