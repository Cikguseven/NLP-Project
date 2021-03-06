bots = ['AmputatorBot', 'AutoModerator', 'bot-amos-counter',
        'Gfycat_Details_Fixer', 'imguralbumbot', 'LearnDifferenceBot',
        'RemindMeBot', 'savevideobot', 'sneakpeek_botGifReversingBot',
        'timestamp_bot', 'TotesMessenger', 'twinkiac', 'VredditDownloader',
        'vredditshare', 'WikiMobileLinkBot', 'WikiTextBot']

uncased_regex_replacements = [
    # Remove links and references to reddit users and subreddits
    (r'(?<![^\W_])http\S+', ''),
    (r'(?<![^\W_])/?u/\S+', ''),
    (r'(?<![^\W_])/?r/\S+', ''),

    # Formatting
    ('&#x200B', ''),
    (r'\‘|\’', '\''),
    (r' & ', ' and '),
    (r' = ', ' equals to '),
    (r'(?<![^\W_])S\$', '$'),
    (r'^\W\s', ''),

    # Remove sentence-final particles
    (r'(?<![^\W_])ah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])ai\s?y(a|o){1,}(?![^\W_])', ''),
    (r'(?<![^\W_])aiy(a|o)h{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])bah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])hah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])har{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])hor{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])lah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])leh{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])lor{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])meh{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])siah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])wah{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])wor{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])lol{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])lo{1,}l(?![^\W_])', ''),
    (r'(?<![^\W_])lmao{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])rofl{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])xd{1,}(?![^\W_])', ''),
    (r'(?<![^\W_])(?:a*(?:ha)+h?|(?:l+o+)+l+)(?![^\W_])', ''),

    # Context specific replacements
    (r'(?<![^\W_])/s(?![^\W_])', ' I am being sarcastic'),
    (r'(?<![^\W_])a(ne| and e|&e)(?![^\W_])', ' a&e'),
    (r'(?<![^\W_])abit(?![^\W_])', ' abit'),
    (r'(?<![^\W_])abt(?![^\W_])', ' about'),
    (r'(?<![^\W_])admins?(?![^\W_])', ' administrator'),
    (r'(?<![^\W_])af(?![^\W_])', ' as fuck'),
    (r'(?<![^\W_])afaik(?![^\W_])', ' as far as i know'),
    (r'(?<![^\W_])aiui(?![^\W_])', ' as I understand it'),
    (r'(?<![^\W_])appt(?![^\W_])', ' appointment'),
    (r'(?<![^\W_])arnd(?![^\W_])', ' around'),
    (r'(?<![^\W_])atm(?![^\W_])', ' atm'),
    (r'(?<![^\W_])b?(c|coz|cos|cuz|cus)(?![^\W_])', ' because'),
    (r'(?<![^\W_])bo\s?(b|p)ian(?![^\W_])', ' no choice'),
    (r'(?<![^\W_])bo\s?ch(a|u)p(?![^\W_])', ' do not care'),
    (r'(?<![^\W_])br(o{1,}|uh{1,})(?![^\W_])', ' brother'),
    (r'(?<![^\W_])bto(?![^\W_])', ' build-to-order'),
    (r'(?<![^\W_])btw(?![^\W_])', ' by the way'),
    (r'(?<![^\W_])btwn(?![^\W_])', ' between'),
    (r'(?<![^\W_])c(oronavirus|ovid-19|ovid)(?![^\W_])', ' covid'),
    (r'(?<![^\W_])c\+(?![^\W_])', ' covid positive'),
    (r'(?<![^\W_])c?cb(?![^\W_])', ' cheebye'),
    (r'(?<![^\W_])ccs(?![^\W_])', ' Chan Chun Sing'),
    (r'(?<![^\W_])cmi(?![^\W_])', ' cannot make it'),
    (r'(?<![^\W_])cpf(?![^\W_])', ' cpf'),
    (r'(?<![^\W_])cpu(?![^\W_])', ' central processing unit'),
    (r'(?<![^\W_])dun(?![^\W_])', " don't"),
    (r'(?<![^\W_])edit(?![^\W_])', ' edit'),
    (r'(?<![^\W_])etf(?![^\W_])', ' exchange-traded fund'),
    (r'(?<![^\W_])egm(?![^\W_])', ' extraordinary general meeting'),
    (r'(?<![^\W_])ems(?![^\W_])', ' emergency medical services'),
    (r'(?<![^\W_])f(ark|k|ck|uk)|knn(?![^\W_])', ' fuck'),
    (r'(?<![^\W_])f(nb| and b|&b)(?![^\W_])', ' f&b'),
    (r'(?<![^\W_])fkin(g)*(?![^\W_])', ' fucking'),
    (r'(?<![^\W_])fml{1,}(?![^\W_])', ' fuck my life'),
    (r'(?<![^\W_])fwiw(?![^\W_])', ' for what it is worth'),
    (r'(?<![^\W_])fyi(?![^\W_])', ' for your information'),
    (r'(?<![^\W_])g(ov|ovt|ahmen|ahment|armen|arment|overnment)(?![^\W_])', ' Government'),
    (r'(?<![^\W_])gamxia(?![^\W_])', ' thank you'),
    (r'(?<![^\W_])gbh(?![^\W_])', ' grievous bodily harm'),
    (r'(?<![^\W_])gky(?![^\W_])', ' Gan Kim Yong'),
    (r'(?<![^\W_])glc(?![^\W_])', ' government-linked company'),
    (r'(?<![^\W_])gonna(?![^\W_])', ' going to'),
    (r'(?<![^\W_])gpu(?![^\W_])', ' graphics processing unit'),
    (r'(?<![^\W_])grc(?![^\W_])', ' group representation constituency'),
    (r'(?<![^\W_])gst(?![^\W_])', ' goods and services tax'),
    (r'(?<![^\W_])gta(?![^\W_])', ' Grand Theft Auto'),
    (r'(?<![^\W_])hcw', ' healthcare worker'),
    (r'(?<![^\W_])herculean(?![^\W_])', ' herculean'),
    (r'(?<![^\W_])hse(?![^\W_])', ' house'),
    (r'(?<![^\W_])icu(?![^\W_])', ' intensive care unit'),
    (r'(?<![^\W_])idk(?![^\W_])', ' I do not know'),
    (r'(?<![^\W_])iirc(?![^\W_])', ' if I remember correctly'),
    (r'(?<![^\W_])ikr(?![^\W_])', ' I know right'),
    (r'(?<![^\W_])imo(?![^\W_])', ' in my opinion'),
    (r'(?<![^\W_])irl(?![^\W_])', ' in real life'),
    (r'(?<![^\W_])itt(?![^\W_])', ' in this thread'),
    (r'(?<![^\W_])jc(?![^\W_])', ' junior college'),
    (r'(?<![^\W_])kiasi(?![^\W_])', ' afraid to die'),
    (r'(?<![^\W_])kiasu(?![^\W_])', ' afraid to lose'),
    (r'(?<![^\W_])kns(?![^\W_])', ' like shit'),
    (r'(?<![^\W_])ktv(?![^\W_])', ' karaoke lounge'),
    (r'(?<![^\W_])lhl(?![^\W_])', ' Lee Hsien Loong'),
    (r'(?<![^\W_])lky(?![^\W_])', ' Lee Kuan Yew'),
    (r'(?<![^\W_])medisave(?![^\W_])', ' medisave'),
    (r'(?<![^\W_])mrt(?![^\W_])', ' MRT'),
    (r'(?<![^\W_])msia(?![^\W_])', ' Malaysia'),
    (r'(?<![^\W_])ngl(?![^\W_])', ' not going to lie'),
    (r'(?<![^\W_])nric(?![^\W_])', ' national registration identity card'),
    (r'(?<![^\W_])nsf(?![^\W_])', ' fulltime national serviceman'),
    (r'(?<![^\W_])nsmen(?![^\W_])', ' national serviceman'),
    (r'(?<![^\W_])nvr(?![^\W_])', ' never'),
    (r'(?<![^\W_])ofc(?![^\W_])', ' of course'),
    (r'(?<![^\W_])omg(?![^\W_])', ' oh my god'),
    (r'(?<![^\W_])omn?icron(?![^\W_])', ' omicron'),
    (r'(?<![^\W_])ord(?![^\W_])', ' operationally ready date'),
    (r'(?<![^\W_])oso(?![^\W_])', ' also'),
    (r'(?<![^\W_])otter(?![^\W_])', ' otter'),
    (r'(?<![^\W_])oyk(?![^\W_])', ' Ong Ye Kung'),
    (r'(?<![^\W_])p(eo)?pl(e)?(?![^\W_])', ' people'),
    (r'(?<![^\W_])pepega(?![^\W_])', ' stupid'),
    (r'(?<![^\W_])pnsf(?![^\W_])', ' fulltime police national serviceman'),
    (r'(?<![^\W_])polis(?![^\W_])', ' police'),
    (r'(?<![^\W_])poly(?![^\W_])', ' polytechnic'),
    (r'(?<![^\W_])ppe(?![^\W_])', ' personal protective equipment'),
    (r'(?<![^\W_])pri(?![^\W_])', ' primary'),
    (r'(?<![^\W_])psle(?![^\W_])', ' psle'),
    (r'(?<![^\W_])religion(?![^\W_])', ' religion'),
    (r'(?<![^\W_])sars(?![^\W_])', ' severe acute respiratory syndrome'),
    (r'(?<![^\W_])sbo(?![^\W_])', ' standard battle order'),
    (r'(?<![^\W_])s(g|\'pore)(?![^\W_])', ' Singapore'),
    (r'(?<![^\W_])s(g|\'po)rean|sinkie(?![^\W_])', ' Singaporean'),
    (r'(?<![^\W_])sdas(?![^\W_])', ' safe distancing ambassadors'),
    (r'(?<![^\W_])sec(?![^\W_])', ' secondary'),
    (r'(?<![^\W_])shn(?![^\W_])', ' SHN'),
    (r'(?<![^\W_])shld(?![^\W_])', ' should'),
    (r'(?<![^\W_])siam(?![^\W_])', ' avoid'),
    (r'(?<![^\W_])sigma(?![^\W_])', ' sigma'),
    (r'(?<![^\W_])sisyphean(?![^\W_])', ' sisyphean'),
    (r'(?<![^\W_])smth(?![^\W_])', ' something'),
    (r'(?<![^\W_])srs(?![^\W_])', ' serious'),
    (r'(?<![^\W_])stds?(?![^\W_])', ' sexually transmitted diseases'),
    (r'(?<![^\W_])stronk(?![^\W_])', ' strong'),
    (r'(?<![^\W_])tbf(?![^\W_])', ' to be fair'),
    (r'(?<![^\W_])tbh(?![^\W_])', ' to be honest'),
    (r'(?<![^\W_])tcj(?![^\W_])', ' Tan Chuan Jin'),
    (r'(?<![^\W_])tmi(?![^\W_])', ' too much information'),
    (r'(?<![^\W_])ttm(?![^\W_])', ' to the maximum'),
    (r'(?<![^\W_])u(?![^\W_])', ' you'),
    (r'(?<![^\W_])ui(?![^\W_])', ' user interface'),
    (r'(?<![^\W_])unclos(?![^\W_])', ' unclos'),
    (r'(?<![^\W_])uni(?![^\W_])', ' university'),
    (r'(?<![^\W_])ux(?![^\W_])', ' user experience'),
    (r'(?<![^\W_])wmd(?![^\W_])', ' weapon of mass destruction'),
    (r'(?<![^\W_])w(rt|\.r\.t\.|\.r\.t)(?![^\W_])', ' with respect to'),
    (r'(?<![^\W_])wtf(?![^\W_])', ' what the fuck'),
    (r'(?<![^\W_])ww(ii|2)(?![^\W_])', ' World War Two'),
    (r'(?<![^\W_])ww(i|1)(?![^\W_])', ' World War One'),
    (r'(?<![^\W_])yolo{1,}(?![^\W_])', ' you only live once'),

    (r'(?<![^\W_])REC(?![^\W_])', ' recruit'),
    (r'(?<![^\W_])PFC(?![^\W_])', ' private'),
    (r'(?<![^\W_])LCP(?![^\W_])', ' lance corporal'),
    (r'(?<![^\W_])CPL(?![^\W_])', ' corporal'),
    (r'(?<![^\W_])CFC(?![^\W_])', ' corporal first class'),
    (r'(?<![^\W_])3SG(?![^\W_])', ' third sergeant'),
    (r'(?<![^\W_])2SG(?![^\W_])', ' second sergeant'),
    (r'(?<![^\W_])1SG(?![^\W_])', ' first sergeant'),
    (r'(?<![^\W_])SSG(?![^\W_])', ' staff sergeant'),
    (r'(?<![^\W_])3WO(?![^\W_])', ' third warrant officer'),
    (r'(?<![^\W_])2WO(?![^\W_])', ' second warrant officer'),
    (r'(?<![^\W_])1WO(?![^\W_])', ' first warrant officer'),
    (r'(?<![^\W_])MWO(?![^\W_])', ' master warrant officer'),
    (r'(?<![^\W_])SWO(?![^\W_])', ' senior warrant officer'),
    (r'(?<![^\W_])2LT(?![^\W_])', ' second lieutenant'),
    (r'(?<![^\W_])CPT(?![^\W_])', ' captain'),
    (r'(?<![^\W_])MAJ(?![^\W_])', ' major'),
    (r'(?<![^\W_])LTC(?![^\W_])', ' lieutenant colonel'),
    (r'(?<![^\W_])SLTC(?![^\W_])', ' senior lieutenant colonel'),
    (r'(?<![^\W_])COL(?![^\W_])', ' colonel'),
    (r'(?<![^\W_])ME1(?![^\W_])', ' military expert 1'),
    (r'(?<![^\W_])ME2(?![^\W_])', ' military expert 2'),
    (r'(?<![^\W_])ME3(?![^\W_])', ' military expert 3'),
    (r'(?<![^\W_])ME4(?![^\W_])', ' military expert 4'),
    (r'(?<![^\W_])ME5(?![^\W_])', ' military expert 5'),
    (r'(?<![^\W_])ME6(?![^\W_])', ' military expert 6'),
    (r'(?<![^\W_])ME7(?![^\W_])', ' military expert 7'),
    (r'(?<![^\W_])ME8(?![^\W_])', ' military expert 8'),

    # Remove repeated characters that have little meaning
    (r'(.)\1{9,}', ''),
    (r'\.{2,}', '.'),
    (r'\?{2,}', '?'),
    (r',{2,}', ','),
    (r'!{2,}', '!'),
    (r'\s{2,}', ' '),
    (r'\/', ' or '),

    # Keep only these characters
    ('[^A-Za-z0-9 .,!?\'/$&%+-]', ' '),

    # Remove additional spaces
    ('\s+', ' ')
]

cased_regex_replacements = [
    (r'(?<![^\W_])IPO(?![^\W_])', " initial public offering"),
    (r"(?<![^\W_])PAP|people's Action Party(?![^\W_])", " People's Action Party"),
    (r'(?<![^\W_])\'?em(?![^\W_])', ' them'),
    (r'(?<![^\W_])ART(?![^\W_])', ' antigen rapid test'),
    (r'(?<![^\W_])C(o|O)D(?![^\W_])', " Call of Duty"),
    (r'(?<![^\W_])EQ(?![^\W_])', ' emotional quotient'),
    (r'(?<![^\W_])IQ(?![^\W_])', ' intelligence quotient'),
    (r'(?<![^\W_])M(o|O)E(?![^\W_])', " Ministry of Education"),
    (r'(?<![^\W_])M(o|O)M(?![^\W_])', " Ministry of Manpower"),
    (r'(?<![^\W_])NS(?![^\W_])', ' national service'),
    (r'(?<![^\W_])RIP(?![^\W_])', ' rest in peace'),
    (r'(?<![^\W_])SEA(?![^\W_])', ' South East Asia'),
    (r'(?<![^\W_])tht(?![^\W_])', ' that'),
    ('\s+', ' '),
]
