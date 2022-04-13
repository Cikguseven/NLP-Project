cased_expressway = {'AYE', 'PIE'}

cased_lgbt = {'LGBT', 'LGBTQ'}

cased_mrt = {'NSL', 'NEL', 'EWL', 'CCL', 'DTL', 'TEL'}

cased_law = {'377A', '377a', 'SHN', 'POFMA'}

cased_products = {'Call of Duty'}

cased_misc = cased_expressway | cased_lgbt | cased_mrt | cased_law | cased_products

cased_sg_org = {'AGO', 'BOA', 'CAB', 'KK', 'NETS', 'NP', 'PA', 'Scoot', 'TP'}

cased_political_parties = {'PAP', 'PSP', 'PPP', 'PV', 'RDU' 'SPP', 'DPP', 'SDP', 'SUP', 'WP', 'SJP', 'PKMS'}

cased_org = cased_sg_org | cased_political_parties

sg_loc = {'adam road', 'admiralty', 'adventure cove', 'airport road', 'alexandra hill', 'alexandra north', 'aljunied', 'amoy street', 'anak bukit', 'anchorvale', 'ang mo kio', 'ann siang hill', 'anson', 'arab street', 'armenian church', 'aviation park', 'baba house', 'bahar junction', 'bahtera', 'bajau', 'bakau', 'balestier', 'bangkit', 'bartley', 'bayfront', 'bayshore', 'beauty world', 'bedok', 'bencoolen', 'bendemeer', 'benoi sector', 'bidadari', 'bishan', 'boat quay', 'boon keng', 'boon lay', 'botanic gardens', 'braddell', 'bras basah', 'brickland', 'brickworks', 'bright hill', 'buangkok', 'buddha tooth relic temple and museum', 'bugis', 'bukit batok', 'bukit brown', 'bukit gombak', 'bukit ho swee', 'bukit merah', 'bukit panjang', 'bukit timah', 'bukom', 'buona vista', 'cairnhill', 'caldecott', 'canberra', 'cantonment', 'cashew', 'cecil', 'central water catchment', 'changi', 'chatsworth', 'cheng lim', 'cheng san', 'chin bee', 'china cultural centre', 'china square', 'chinatown', 'chinese garden', 'choa chu kang', 'chua chu kang', 'chong boon', 'city hall', 'clarke quay', 'clementi', 'clifford pier', 'club street', 'commonwealth', 'compassvale', 'coney island', 'coral edge', 'coronation road', 'crawford', 'dairy farm', 'dakota', 'damai', 'defu', 'depot road', 'dhoby ghaut', 'dieppe barracks', 'dover', 'downtown', 'dunearn', 'east coast', 'elias', 'empress place', 'enterprise', 'eunos', 'everton park', 'expo', 'fajar', 'farmway', 'farrer court', 'farrer park', 'farrer road', 'fernvale', 'flora drive', "foo hai ch'an monastery", "founders' memorial", 'frankel avenue', 'gali batu', 'gedong', 'gek poh', 'geylang', 'ghim moh', 'gombak', 'goodwood park', 'great world', 'greenwood park', 'guilin', 'guillemard', 'gul basin', 'gul circle', 'harbourfront', 'havelock', 'helix bridge', 'henderson waves', 'hillcrest', 'hillview', 'holland drive', 'holland road', 'holland village', 'hong kah', 'hong san see', 'hougang', 'house of tan yeok nee', 'hume', 'institution hill', 'international business park', 'jalan besar', 'jalan kayu', 'jelapang', 'jelebu', 'joo chiat', 'joo koon', 'joo seng', 'jurong bird park', 'jurong', 'kadaloor', 'kaki bukit', 'kallang', 'kallang bahru', 'kampong glam', 'kampong java', 'kangkar', 'kapal', 'katong', 'keat hong', 'kebun baru', 'kembangan', 'kent ridge', 'keppel', 'khatib', 'kian teck', 'kim keat', 'king albert park', 'kovan', 'kranji', 'kupang', 'kusu island', 'labrador park', 'lakeside', 'lavender', 'layar', 'leedon park', 'lentor', 'leonie hill', 'lim chu kang', 'little india', 'lorong ah soo', 'lorong chuan', 'lorong danau', 'lorong halus', 'lorong halus north', 'lower seletar', 'loyang', 'loyang east', 'loyang west', 'macpherson', 'maghain aboth synagogue', 'mandai', 'mandai east', 'mandai estate', 'mandai west', 'margaret drive', 'marina bay', 'marina centre', 'marina east', 'marina south', 'marina south pier', 'marine parade', 'marine terrace', 'marsiling', 'marymount', 'mattar', 'maxwell', 'mayflower', 'mei chin', 'meridian', 'midview', "monk's hill", 'moulmein', 'mount emily', 'mount faber', 'mount pleasant', 'mountbatten', 'murai', 'murai north', 'mutf', 'nanyang bridge', 'nanyang crescent', 'nanyang gateway', 'napier', 'nassim', 'nature reserve', 'nee soon', 'nee soon driclad', 'newton', 'newton circus', 'nibong', 'nicoll highway', 'north coast', 'north-eastern islands', 'northland', 'northshore', 'novena', 'oasis', 'one north', 'one tree hill', 'one-north', 'orange grove', 'orchard', 'orchard boulevard', 'orchard road', 'outram', 'outram park', 'oxley', 'palawan beach', 'pandan', 'pandan reservoir', 'pang sua', 'pasir laba', 'pasir panjang', 'pasir ris', 'pasir ris central', 'pasir ris drive', 'pasir ris east', 'pasir ris park', 'pasir ris wafer fab park', 'pasir ris west', 'paterson', 'paya lebar', 'paya lebar east', 'paya lebar north', 'paya lebar west', "pearl's hill", 'pending', 'peng kang hill', 'peng siang', 'penjuru crescent', "people's park", 'petir', 'phoenix', 'pioneer', 'pioneer sector', 'poh chiak keng', 'potong pasir', 'poyan', 'prince edward road', 'promenade', 'pulau punggol barat', 'pulau punggol timor', 'pulau seletar', 'punggol', 'punggol canal', 'punggol coast', 'punggol field', 'punggol point', 'queenstown', 'queensway', 'raffles hotel', 'raffles place', 'ranggung', 'redhill', 'renjong', 'reservoir view', 'ridout', 'river valley', 'rivervale', 'riviera', 'robertson quay', 'rochor', 'rochor canal', 'rumbia', 'sam kee', 'samudera', 'samulun', 'sarimbun', 'saujana', 'segar', 'selegie', 'seletar', 'seletar aerospace park', 'seletar hills', 'semakau', 'sembawang', 'sembawang central', 'sembawang east', 'sembawang hills', 'sembawang north', 'sembawang spring', 'sembawang straits', 'sembawang wharves', 'sengkang', 'sengkang town centre', 'sengkang west', 'senja', 'sennett', 'senoko north', 'senoko south', 'senoko west', 'sentosa', 'serangoon', 'serangoon central', 'serangoon garden', 'serangoon north', 'shenton way', 'shipyard', 'siglap', 'siloso beach', 'simei', 'simpang', 'simpang bedok', 'simpang north', 'simpang south', 'singapore river', "sisters' island", 'sixth avenue', 'somerset', 'soo teck', 'south view', 'southern islands', 'southern ridges', 'springleaf', 'sri temasek', 'st john island', 'stadium', 'stevens', 'sudong', 'sultan mosque', 'sumang', 'sungei bedok', 'sungei gedong', 'sungei kadut', 'sungei road', 'sunset way', 'swiss club', 'tagore', 'tai seng', 'taman jurong', 'tampines', 'tampines east', 'tampines north', 'tampines west', 'tan kah kee', 'tanah merah', 'tanglin', 'tanglin halt', 'tanjong irau', 'tanjong katong', 'tanjong pagar', 'tanjong rhu', 'tavistock', 'tawas', 'teban gardens', 'teck ghee', 'teck lee', 'teck whye', 'telok ayer', 'telok blangah', 'telok blangah drive', 'telok blangah rise', 'telok blangah way', 'telok kurau', 'ten mile junction', 'tengah', 'tengah park', 'tengah plantation', 'tengeh', 'thanggam', 'thomson', 'thousand oaks', 'tiong bahru', 'toa payoh', 'toa payoh central', 'toa payoh west', 'toh guan', 'toh tuck', 'tongkang', 'townsville', 'tuas', 'tuas bay', 'tuas crescent', 'tuas link', 'tuas north', 'tuas promenade', 'tuas view', 'tuas view extension', 'tuas west road', 'tukang', 'turf club', 'tyersall', 'ubi', 'ulu pandan', 'upper changi', 'upper paya lebar', 'upper seletar', 'upper thomson', 'ura city gallery', 'waterloo street', 'waterway point', 'west coast', 'western islands', 'western water catchment', 'woodgrove', 'woodlands', 'woodlands east', 'woodlands north', 'woodlands regional centre', 'woodlands south', 'woodlands west', 'woodleigh', 'wrexham', 'xilin', 'yew tee', 'yio chu kang', 'yio chu kang east', 'yio chu kang north', 'yio chu kang west', 'yishun', 'yishun central', 'yishun east', 'yishun south', 'yishun west', 'yuhua east', 'yuhua west', 'yunnan'}

sg_org = {'m1', 'ri', 'sq', 'acs', 'agc', 'ava', 'bca', 'cag', 'ccs', 'cea', 'cgh', 'cjc', 'cnb', 'cpe', 'cra', 'csa', 'csc', 'dbs', 'dhs', 'edb', 'ejc', 'ema', 'esg', 'f&n', 'hci', 'hdb', 'hlb', 'hpb', 'hsa', 'hta', 'htx', 'iac', 'ica', 'imh', 'isd', 'ite', 'jtc', 'lsb', 'lta', 'mas', 'mci', 'mda', 'mfa', 'mgs', 'mha', 'mnd', 'mof', 'moh', 'mot', 'mpa', 'mse', 'msf', 'mti', 'nac', 'ndc', 'nea', 'nhb', 'nhg', 'njc', 'nlb', 'nol', 'nsc', 'ntu', 'nuh', 'nus', 'nyc', 'ocs', 'peb', 'pmo', 'psa', 'psc', 'ptc', 'pub', 'rgs', 'sac', 'sal', 'sbs', 'scb', 'scs', 'sdc', 'sfa', 'sgh', 'sia', 'sit', 'sji', 'sla', 'smc', 'smu', 'snb', 'spc', 'spf', 'sph', 'sss', 'sst', 'stb', 'tjc', 'tpg', 'ttc', 'uob', 'ura', 'vjc', 'wsg', '2c2p', 'acjc', 'acra', 'acsi', 'ahtc', 'aibi', 'bmtc', 'caas', 'cccs', 'chij', 'cpib', 'csit', 'dsta', 'ecda', 'ectc', 'fstd', 'grab', 'imda', 'ipos', 'iras', 'jbtc', 'jpjc', 'jrtc', 'ktph', 'mccy', 'mptc', 'mshs', 'muis', 'nafa', 'nccs', 'nchs', 'ncid', 'ncpc', 'nptd', 'nstc', 'ntuc', 'nuhs', 'nygh', 'nyjc', 'ocbc', 'osim', 'pdpc', 'posb', 'rvhs', 'sajc', 'sats', 'sbtc', 'scdf', 'scgs', 'seab', 'sgag', 'shps', 'sktc', 'smrt', 'snec', 'sngs', 'sota', 'tkgs', 'tmjc', 'tptc', 'ttsh', 'wctc', 'yijc', 'amktc', 'asrjc', 'astar', 'btptc', 'ccktc', 'cisco', 'dnata', 'hbptc', 'ij tp', 'kcpss', 'koufu', 'myttc', 'ntfgh', 'plmgs', 'qoo10', 'score', 'tangs', 'tcmpb', 'zero1', 'a*star', 'acs(i)', 'aspial', 'garena', 'hyflux', 'istana', 'lazada', 'mindef', 'minlaw', 'nparks', 'prpgtc', 'redone', 'shopee', 'vivifi', 'wilmar', 'antlabs', 'chij tp', 'govtech', 'lasalle', 'silkair', 'singtel', 'sportsg', 'starhub', 'tao nan', 'ascendas', 'broadcom', 'cat high', 'dbs bank', 'deyi sec', 'filmtack', 'fjcourts', 'nus high', 'qifa pri', 'sembcorp', 'singpost', 'sp group', 'st hilda', 'supcourt', 'bowen sec', 'breadtalk', 'carousell', 'cedar pri', 'coral pri', 'cpf board', 'crest sec', 'damai pri', 'damai sec', 'eunos pri', 'fairprice', 'fajar sec', 'fuhua pri', 'fuhua sec', 'mediacorp', 'oasis pri', 'qihua pri', 'robinsons', 'sing post', 'st andrew', 'st joseph', 'st. hilda', 'unity pri', 'unity sec', 'yuhua pri', 'yuhua sec', 'yumin pri', 'ayam brand', 'beacon pri', 'beatty sec', 'capitaland', 'dunman sec', 'fuchun pri', 'fuchun sec', 'globalroam', 'hua yi sec', 'huamin pri', 'innova pri', 'jiemin pri', 'jurong pri', 'jurong sec', 'juying pri', 'juying sec', 'keming pri', 'kranji pri', 'kranji sec', 'loyang pri', 'myrepublic', 'outram sec', 'peicai sec', 'peirce sec', 'regent sec', 'rulang pri', 'shuqun pri', 'shuqun sec', 'st anthony', 'st gabriel', 'st patrick', 'st stephen', 'st. andrew', 'st. joseph', 'stan chart', 'tote board', 'valour pri', 'xinmin pri', 'xinmin sec', 'xishan pri', 'yishun pri', 'yishun sec', 'yuying sec', 'angsana pri', 'bartley sec', 'chung cheng', 'concord pri', 'da qiao pri', 'dazhong pri', 'dunearn sec', 'flextronics', 'horizon pri', 'hougang pri', 'hougang sec', 'junyuan pri', 'junyuan sec', 'jurong port', 'lianhua pri', 'nan hua pri', 'nanyang pri', 'pei hwa sec', 'peiying pri', 'ping yi sec', 'pioneer pri', 'punggol pri', 'punggol sec', 'saint hilda', 'sbs transit', 'sheng siong', 'si ling pri', 'spectra sec', 'st margaret', 'st. anthony', 'st. gabriel', 'st. patrick', 'st. stephen', 'tanglin sec', 'tee yih jia', 'temasek pri', 'temasek sec', 'the cabinet', 'tpg telecom', 'whitley sec', 'xinghua pri', 'xingnan pri', 'yew tee pri', 'yu neng pri', 'zhangde pri', 'anderson pri', 'anderson sec', 'boon lay sec', 'canberra pri', 'canberra sec', 'changkat pri', 'circles.life', 'clementi pri', 'fengshan pri', 'fernvale pri', 'frontier pri', 'hong kah sec', 'lakeside pri', 'manjusri sec', 'meridian pri', 'meridian sec', 'montfort sec', 'new town pri', 'new town sec', 'ngee ann pri', 'ngee ann sec', 'pei tong pri', 'quest global', 'saint andrew', 'saint joseph', 'sengkang sec', 'smart nation', 'st. margaret', 'stamford pri', 'state courts', 'tampines pri', 'tampines sec', 'waterway pri', 'westwood pri', 'westwood sec', 'zhenghua pri', 'zhenghua sec', 'zhonghua pri', 'zhonghua sec', 'admiralty pri', 'admiralty sec', 'alexandra pri', 'bendemeer pri', 'bendemeer sec', 'broadrick sec', 'casuarina pri', 'chij (katong)', 'comfortdelgro', 'east view pri', 'edgefield pri', 'edgefield sec', 'endeavour pri', 'evergreen pri', 'evergreen sec', 'gongshang pri', 'greendale pri', 'greendale sec', 'greenwood pri', 'guangyang pri', 'guangyang sec', 'hillgrove sec', 'jing shan pri', 'marsiling pri', 'marsiling sec', 'mayflower pri', 'mayflower sec', 'nan chiau pri', 'northland pri', 'northland sec', 'northoaks pri', 'palm view pri', 'park view pri', 'pasir ris pri', 'pasir ris sec', 'queensway sec', 'radin mas pri', 'raffles girls', 'riverside pri', 'riverside sec', 'rivervale pri', 'rosyth school', 'saint anthony', 'saint gabriel', 'saint patrick', 'saint stephen', 'sembawang pri', 'sembawang sec', 'seng kang pri', 'serangoon sec', 'teck ghee pri', 'teck whye pri', 'teck whye sec', 'tiger airways', 'west view pri', 'woodgrove pri', 'woodgrove sec', 'woodlands pri', 'woodlands sec', 'yangzheng pri', 'yeo hiap seng', 'ai tong school', 'ang mo kio pri', 'ang mo kio sec', 'bedok view sec', 'bukit view pri', 'bukit view sec', 'cantonment pri', 'chij (kellock)', 'chongfu school', 'chongzheng pri', 'east coast pri', 'elias park pri', 'greenridge pri', 'greenridge sec', 'henry park pri', 'kent ridge sec', 'macpherson pri', 'mee toh school', 'naval base pri', 'naval base sec', 'north view pri', 'queenstown pri', 'queenstown sec', 'saint margaret', 'singapore army', 'singapore post', 'south view pri', 'springdale pri', 'st engineering', 'surbana jurong', 'townsville pri', 'wellington pri', 'west grove pri', 'yuan ching sec', 'bedok green pri', 'bedok green sec', 'bedok south sec', 'bee cheng hiang', 'bukit batok sec', 'bukit merah sec', 'bukit timah pri', 'charles & keith', 'compassvale pri', 'compassvale sec', 'corporation pri', 'east spring pri', 'east spring sec', 'farrer park pri', 'hong wen school', 'jtc corporation', 'jurong west pri', 'jurong west sec', 'jurongville sec', 'kong hwa school', 'loyang view sec', 'ministry of law', 'north vista pri', 'north vista sec', 'northbrooks sec', 'orchid park sec', 'organs of state', 'singapore power', 'sport singapore', 'springfield sec', 'telok kurau pri', 'trafigura group', 'twelve cupcakes', 'victoria school', 'west spring pri', 'west spring sec', 'white sands pri', 'yishun town sec', 'yusof ishak sec', 'anchor green pri', 'blangah rise pri', "cedar girls' sec", 'commonwealth sec', 'far east orchard', 'fraser and neave', 'gan eng seng pri', 'ite college east', 'ite college west', 'north spring pri', 'opera estate pri', 'poi ching school', 'popular holdings', 'punggol cove pri', 'punggol view pri', 'river valley pri', 'spring singapore', 'temasek holdings', 'yio chu kang pri', 'yio chu kang sec', 'ahmad ibrahim pri', 'ahmad ibrahim sec', 'bukit panjang pri', 'christ church sec', 'chua chu kang pri', 'chua chu kang sec', 'clementi town sec', 'genting singapore', 'home team academy', 'maha bodhi school', 'punggol green pri', 'raffles girls pri', 'swiss cottage sec', 'ya kun kaya toast', 'alexandra hospital', 'balestier hill pri', 'boustead singapore', 'china aviation oil', 'de la salle school', 'dunman high school', 'great eastern life', "haig girls' school", 'jurong consultants', 'keppel corporation', 'kheng cheng school', 'ministry of health', 'school of the arts', 'sengkang green pri', 'singapore airlines', 'singapore exchange', 'standard chartered', 'tampines north pri', 'tanjong katong pri', 'tanjong katong sec', 'woodlands ring pri', 'woodlands ring sec', 'board of architects', 'boon lay garden pri', 'changkat changi sec', 'chij katong convent', 'creative technology', 'first toa payoh pri', 'gan eng seng school', "holy innocents' pri", 'ite college central', 'ministry of defence', 'ministry of finance', 'nan hua high school', 'nanyang polytechnic', 'pasir ris crest sec', 'raffles institution', 'red swastika school', 'sikh advisory board', 'temasek polytechnic', 'thakral corporation', 'workforce singapore', 'anglican high school', 'anglo-chinese school', 'canossa catholic pri', 'catholic high school', 'changi airport group', 'elections department', 'enterprise singapore', 'hindu advisory board', 'jetstar asia airways', 'land surveyors board', 'ministry of manpower', 'national parks board', 'national skin centre', 'neptune orient lines', 'ngee ann polytechnic', 'officer cadet school', "people's association", 'raffles girls school', 'republic polytechnic', 'science centre board', 'serangoon garden sec', 'civil service college', 'cyber security agency', 'eunoia junior college', 'far east organization', 'golden agri-resources', 'hwa chong institution', 'ministry of education', 'ministry of transport', 'nan chiau high school', 'national arts council', 'nee soon town council', 'sengkang town council', 'singapore food agency', 'singapore polytechnic', 'tampines town council', "crescent girls' school", 'health promotion board', 'hindu endowments board', 'hotels licensing board', 'montfort junior school', 'nanyang junior college', 'national library board', 'national youth council', 'pei chun public school', 'princess elizabeth pri', 'public utilities board', 'sembawang town council', 'singapore armed forces', 'singapore police force', 'singhealth polyclinics', 'skillsfuture singapore', 'tan tock seng hospital', 'temasek junior college', 'ang mo kio town council', 'catholic junior college', 'changi general hospital', 'designsingapore council', 'east coast town council', 'energy market authority', 'fairfield methodist sec', 'judiciary, state courts', 'jurong town corporation', 'khoo teck puat hospital', "methodist girls' school", 'national computer board', 'national heritage board', 'national junior college', 'parliament of singapore', "prime minister's office", 'public works department', 'san yu adventist school', 'singapore harbour board', 'singapore nursing board', 'singapore sports school', 'singapore tourism board', 'specialist cadet school', 'tcm practitioners board', 'trade development board', 'vertex venture holdings', 'victoria junior college', 'west coast town council', 'woodlands health campus', 'yellow ribbon singapore', "auditor-general's office", 'central narcotics bureau', 'chij primary (toa payoh)', 'geylang methodist school', 'geylang methodist school', 'hai sing catholic school', 'jalan besar town council', 'judiciary, supreme court', 'land transport authority', 'maris stella high school', 'marymount convent school', 'ministry of home affairs', 'pei hwa presbyterian pri', 'post office savings bank', 'presbyterian high school', 'public transport council', 'river valley high school', 'science centre singapore', 'singapore academy of law', 'singapore dental council', 'singapore land authority', 'singapore press holdings', 'singapore prison service', 'singapore sports council', 'aetos security management', 'assumption english school', "chij st. joseph's convent", 'health sciences authority', 'national healthcare group', 'public service commission', 'singapore health services', 'singapore medical council', 'chij secondary (toa payoh)', "chij st. theresa's convent", 'chua chu kang town council', 'economic development board', 'fairfield methodist school', 'institute of mental health', 'kuo chuan presbyterian pri', 'kuo chuan presbyterian sec', 'marine parade town council', "nanyang girls' high school", 'singapore general hospital', 'singapore pharmacy council', 'tanjong pagar town council', "attorney-general's chambers", 'council for estate agencies', 'emergency preparedness unit', "holy innocents' high school", 'iseas–yusof ishak institute', 'jurong port private limited', 'media development authority', 'ministry of foreign affairs', 'national environment agency', 'port of singapore authority', 'river valley junior college', 'saint andrew junior college', 'singapore labour foundation', 'singapore totalisator board', 'anglo-chinese junior college', 'central provident fund board', 'chij our lady queen of peace', 'government technology agency', 'internal security department', 'madrasah alsagoff al-arabiah', 'majlis ugama islam singapura', 'military security department', 'nanyang academy of fine arts', 'national integration council', 'national university hospital', 'renewable energy corporation', 'saint andrews junior college', 'singapore gamma knife centre', "tanjong katong girls' school", 'workforce development agency', 'yishun innova junior college', 'bishan toa payoh town council', 'chij our lady of good counsel', 'chij our lady of the nativity', 'council for private education', 'housing and development board', 'jurong pioneer junior college', 'madrasah aljunied al-islamiah', 'ng teng fong general hospital', "saint andrew's junior college", 'singapore bicentennial office', 'singapore civil defence force', 'singapore national eye centre', 'urban redevelopment authority', 'basic military training centre', 'jurong - clementi town council', 'madrasah al-maarif al-islamiah', 'ministry of trade and industry', 'national archives of singapore', 'aljunied - hougang town council', "chij st. nicholas girls' school", 'madrasah al-arabiah al-islamiah', 'monetary authority of singapore', 'national heart centre singapore', 'national neuroscience institute', 'national university polyclinics', 'preservation of monuments board', 'sentosa development corporation', "singapore chinese girls' school", 'singapore chinese girls’ school', 'singapore management university', 'institute of technical education', 'judiciary, family justice courts', 'madrasah wak tanjong al-islamiah', 'marsiling - yew tee town council', 'ministry of national development', 'nanyang technological university', 'national cancer centre singapore', 'national dental centre singapore', 'national university of singapore', 'pasir ris - punggol town council', 'school of science and technology', 'singapore accountancy commission', 'singapore broadcasting authority', 'systems on silicon manufacturing', 'tampines meridian junior college', 'anderson serangoon junior college', 'madrasah irsyad zuhri al-islamiah', 'national crime prevention council', 'national university health system', 'singapore institute of technology', 'early childhood development agency', "kk women's and children's hospital", 'national council of social service', "paya lebar methodist girls' school", 'singapore broadcasting corporation', 'building and construction authority', 'competition commission of singapore', 'national climate change secretariat', 'national council against drug abuse', 'personal data protection commission', 'singapore petroleum company limited', 'bukit panjang government high school', 'holland - bukit panjang town council', 'infocomm media development authority', 'singapore telecommunications limited', 'civil aviation authority of singapore', 'defence science and technology agency', 'immigration and checkpoints authority', 'inland revenue authority of singapore', 'national healthcare group polyclinics', 'pacific century regional developments', 'corrupt practices investigation bureau', 'home team science and technology agency', 'judiciary, industrial arbitration court', 'national centre for infectious diseases', 'national population and talent division', 'professional engineers board, singapore', 'singapore university of social sciences', 'casino regulatory authority of singapore', 'maritime and port authority of singapore', 'ministry of culture, community and youth', 'telecommunication authority of singapore', 'future systems and technology directorate', 'intellectual property office of singapore', 'ministry of social and family development', 'centre for strategic infocomm technologies', 'ministry of communications and information', 'presidential council for religious harmony', 'agency for science, technology and research', 'singapore examinations and assessment board', 'accounting and corporate regulatory authority', 'singapore university of technology and design', 'commercial and industrial security corporation', 'ministry of sustainability and the environment', 'agri-food and veterinary authority of singapore', 'competition and consumer commission of singapore', 'singapore corporation of rehabilitative enterprises', 'national fire prevention and civil emergency preparedness council'}

political_parties = {"peoples voice", "people's action party", "workers' party", "singapore people's party", "singapore democratic party", "national solidarity party", "reform party", "red dot united", "singapore democratic alliance", "democratic progressive party", "people's power party", "singapore united party", "singapore justice party", "pertubuhan kebangsaan melayu singapura", "singapore malay national organisation"}

all_org = sg_org | political_parties

sg_facs = {'adventure cove waterpark', 'south east asia aquarium', 'amoy quee camp', 'artscience museum', 'burmese buddhist temple', 'causeway', 'changi airport', 'chong pang camp', 'depot road camp', 'esplanade', 'fort canning', 'fort siloso', 'gardens by the bay', 'gbtb', 'haw par villa', 'hendon camp', 'istana', 'jurong bird park', 'keat hong camp', 'khatib camp', 'kong meng san phor kark see monastery', 'kranji camp', 'kwan im thong hood cho temple', 'ladang camp', 'lian shan shuang lin monastery', 'maju camp', 'mandai hill camp', 'marina barrage', 'marina bay sands', 'mbs', 'mowbray camp', 'murai camp', 'nee soon camp', 'night safari', 'pasir laba camp', 'pasir ris camp', 'paya lebar air base', 'resorts world sentosa', 'rifle range road camp', 'river safari', 'rocky hill camp', 'rws', 'safti', 'safti city', 'safti military institute', 'second link', 'selarang camp', 'seletar airport', 'seletar camp', 'sembawang air base', 'sembawang camp', 'singapore discovery centre', 'singapore flyer', 'singapore indoor stadium', 'singapore zoo', 'sri lankaramaya buddhist temple', 'sri mariamman temple', 'sri thendayuthapani temple', 'stagmont camp', 'sun yat sen nanyang memorial hall', 'sungei buloh wetland reserve', 'sungei gedong camp', 'tan si chong su', 'tengah air base', 'thian hock keng', 'tuas checkpoint', 'tuas naval base', 'universal studios singapore', 'victoria theatre and concert hall', 'wat ananda metyarama thai buddhist temple', 'wild wild wet', 'woodlands checkpoint', 'yueh hai ching temple'}

mrt = {'north south line', 'north east line', 'north-south line', 'north-east line', 'circle line', 'downtown line', 'thomson east coast line', 'thomson-east coast line', 'east-west line', 'east west line', 'cross island line', 'cross-island line', 'jurong regional line', 'jurong region line', 'punggol lrt', 'sengkang lrt', 'bukit panjang lrt', 'bplrt'}

expressways = {'pan island expressway', 'ayer rajah expressway', 'north–south corridor', 'north south corridor', 'east coast parkway', 'central expressway', 'tampines expressway', 'kallang-paya lebar expressway', 'kallang paya lebar expressway', 'seletar expressway', 'bukit timah expressway', 'kranji expressway', 'marina coastal expressway', 'bukit timah road', 'jurong island highway', 'nicoll highway', 'west coast highway', 'nsc', 'ecp', 'cte', 'tpe', 'kpe', 'sle', 'bke', 'kje', 'mce'}

religions = {"baha'i", "bahai", "brahma kumaris", "buddhism", "buddhist", "caodaism", "cheondoism", "christian", "christian science", "christianity", "confucianism", "druze", "eckankar", "falun gong", "folk religion", "hare krishna", "hindu", "hinduism", "hoahaoism", "islam", "jainism", "jew", "judaism", "korean shamanism", "latter-day saints", "latter day saints", "mata amritanandamayi math", "muslim", "nichiren shōshū", "quan yin famen", "religion", "sathya sai baba movement", "shinnyo-en", "shinnyo en", "shinto", "shintoism", "sikhism", "soka", "soka gakkai", "spiritism", "taoism", "tenriism", "transcendental meditation", "true jesus church", "tzu chi", "vodou", "world mission society church of god", "zoroastrianism"}

movies = {'agga', 'abtm'}

products = {'geforce', 'rtx'}

events = {"7 month", "7th month", "chinese new year", "chingay", "christmas day", "deepavali", "dragonboat festival", "easter", "good friday", "hari raya", "hungry ghost", "labour day", "lunar new year", "may day", "mid autumn", "national day parade", "national day", "new year", "new year's day", "qingming", "seventh month", "vesak day"}

systems = {'safe entry', 'tracetogether', 'safeentry', 'trace together'}

law = {'unclos'}

all_misc = sg_facs | mrt | expressways | religions | movies | products | events | systems | law

not_ents = {"covid", "doctor", "doctors", "dr", "swab", "nurse", "nurses", "a&e", "virus", "healthcare", "hospital", "wards", "pandemic", "vaccination", "booster", "masks", 'gp', 'supermarket', 'angle', 'ari', 'art', 'cb', 'ccf', 'ctf', 'hbl', 'hra', 'hrp', 'hrw', 'icu', 'pcr', 'phpc', 'qo', 'rrt', 'rsc', 'sash', 'smm', 'vtl', 'wfh', 'liao', 'hdb flat', 'cpf funds', 'oppa', 'kena', 'religion', 'sian'}
