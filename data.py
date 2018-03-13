"""Data to populate the webpage with."""  # flake8: noqa

from collections import namedtuple


_Position = namedtuple(
    "Experience", ["image", "company", "link", "title", "location", "date", "bullets"])

POSITIONS = [
    _Position(
        "polyai.jpg", "PolyAI", "http://www.poly-ai.com/",
        "Lead Scientist & Head of PolyAI Pacific", "Singapore",
        "Jan 2018 - present",
        (
            "Head of the Singapore office",
            "Building a scalable machine learning platform for conversational agents",
        )),
    _Position(
        "carousell.jpg", "Carousell", "http://careers.carousell.com/about/",
        "Principal Data Scientist", "Singapore", "July 2017 - Jan 2018",
        (
            "Built and launched machine-learning features such as "
            "<a href=\"https://medium.com/carousell-insider/deep-image-understanding-at-carousell-121857434837\" target=\"_blank\">"
            "title and category suggestions given image</a>, "
            "<a href=\"https://medium.com/carousell-insider/introducing-chat-reply-on-carousell-3efdf06a72e1\" target=\"_blank\">"
            "chat reply suggestions, and question answering</a>",
            "Lead a team of ML engineers",

        )),
    _Position(
        "google.jpg", "Google", "https://research.google.com/pubs/MatthewHenderson.html",
        "Senior Software Engineer", "Mountain View, California", "Mar 2015 - June 2017",
        (
            "Lead researcher behind Smart Reply in Gmail",
            "Tech lead on "
            "<a href=\"https://en.wikipedia.org/wiki/Ray_Kurzweil\" target=\"_blank\">Ray Kurzweil's</a>"
            " natural language understanding & dialog research team in Mountain View California",
            "Worked for first 6 months in London on text-to-speech research team with "
            "<a href=\"https://research.google.com/pubs/HeigaZen.html\" target=\"_blank\">Heiga Zen</a>"

        )),
    _Position(
        "cambridge.jpg", "University of Cambridge",
        "http://mi.eng.cam.ac.uk/research/dialogue", "PhD", None,
        "Oct 2011 - Feb 2015",
        (
            """
            Supervised by
            <a href="http://mi.eng.cam.ac.uk/~sjy/" title="Steve Young's homepage" target="_blank">Steve Young</a>
            in the
            <a href="http://mi.eng.cam.ac.uk/research/dialogue/" title="Dialogue System's Group" target="_blank">CUED Dialogue Systems Group</a>
            """,
            """
            Funded as a
            <a href="http://research.google.com/university/relations/doctoral_fellowships_europe.html" target="_blank" title="Google Doctoral Fellowship Programme">Google Doctoral Fellow</a>
            in Speech Technology, mentored by
            <a href="http://www.seas.ucla.edu/spapl/bps/">Brian Strope</a>
            """,
            """
            Lead organiser of the
            <a href="http://camdial.org/~mh521/dstc/" target="_blank">Dialog State Tracking Challenges</a>
            """,
            """
            Contributed to VocalIQ,
            <a href="http://www.wsj.com/articles/apple-acquires-artificial-intelligence-startup-vocaliq-1443815801" target="_blank">later acquired by Apple</a>.
            """)),
]


_Publication = namedtuple(
    "Publication", ["title", "subtitle", "date", "description", "pdf", "bibtex"])

PUBLICATIONS = [
    _Publication(
        "Question-Answer Selection in User to User Marketplace Conversations",
        "Girish Kumar, Matthew Henderson, Shannon Chan, Hoang Nguyen, and Lucas Ngoo",
        "2018",
        """
        This paper presents a question answering system, that selects sentences from
        product descriptions using a neural-network ranking model. This is trained
        and evaluated on a dataset of 590K questions and answers from Carousell marketplace
        conversations.
        """,
        "https://arxiv.org/pdf/1802.01766",
        """
@inproceedings{Kumar2018,
    author = {Kumar, Girish and Henderson, Matthew and Chan, Shannon and Nguyen, Hoang and Ngoo, Lucas},
    title = "{Question-Answer Selection in User to User Marketplace Conversations}",
    booktitle = {International Workshop on Spoken Dialog Systems (IWSDS)},
    month = {May},
    year = {2018}
}
        """
    ),
    _Publication(
        "Efficient Natural Language Response Suggestion for Smart Reply",
        "Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun-hsuan Sung, Laszlo Lukacs, Ruiqi Guo, Sanjiv Kumar, Balint Miklos, and Ray Kurzweil",
        "2017",
        """
        The modelling approach behind the Smart Reply feature launched to Gmail, using hierarchical
        feedforward neural networks, and an efficient dot-product search. The system is 100x times
        faster than the original LSTM sequence-to-sequence approach and is higher quality.
        """,
        "https://arxiv.org/pdf/1705.00652",
        """
@article{Henderson2017,
   author = {Henderson, Matthew and {Al-Rfou}, Rami and Strope, Brian and Sung, Yun-hsuan and
	 Luk{\'{a}}cs, L{\'{a}}szl{\'{o}} and Guo, Ruiqi and Kumar, Sanjiv and Miklos, Balint and
	Kurzweil, Ray},
    title = "{Efficient Natural Language Response Suggestion for Smart Reply}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1705.00652},
 primaryClass = "cs.CL",
     year = 2017,
    month = may,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170500652H},
}
        """),
    _Publication(
        "Machine Learning for Dialog State Tracking: A Review",
        "Matthew Henderson",
        "2015",
        """
        A review paper on Machine Learning methods for Dialog State Tracking. An
        invited talk, presented at the first Machine Learning in Spoken Language
        Processing workshop in Aizuwakamatsu, Japan.
        """,
        "https://research.google.com/pubs/archive/44018.pdf",
        """
@inproceedings{Henderson2015,
    title = {Machine Learning for Dialog State Tracking: A Review},
    author  = {Matthew Henderson},
    year  = 2015,
    booktitle = {The First International Workshop on Machine Learning in Spoken Language Processing}
}
        """
    ),
    _Publication(
        "Discriminative Methods for Statistical Spoken Dialogue Systems",
        "PhD Thesis",
        "2015",
        """
        This thesis presents how discriminative machine learning methods can be
        used to develop high accuracy spoken language understanding and
        state tracking modules for spoken dialogue systems.
        """,
        "assets/pdf/Discriminative_Methods_for_Statistical_Spoken_Dialogue_Systems_Matthew_Henderson_PhD_Thesis.pdf",
        """
@phdthesis{MattHendersonThesis,
    Title = {{Discriminative Methods for Statistical Spoken Dialogue Systems}},
    Author = {Henderson, Matthew},
    Year = {2015},
    School = {{University of Cambridge}},
}
        """
    ),
    _Publication(
        "The Third Dialog State Tracking Challenge",
        "Matthew Henderson, Blaise Thomson, and Jason Williams",
        "2014",
        """
        The Third Dialog State Tracking Challenge, with 7 research teams submitting 26 total entries,
        evaluated the state of the art in dialog state tracking. The focus was on adapting to expanding
        conversational domains. Presented as a poster at IEEE SLT 2014, in Lake Tahoe.
        """,
        "assets/pdf/The_Third_Dialog_State_Tracking_Challenge.pdf",
        """
@inproceedings{Henderson2014c,
    author = {Henderson, M. and  Thomson, B. and Williams, J.},
    booktitle = {Proceedings of IEEE Spoken Language Technology},
    title = {{The Third Dialog State Tracking Challenge}},
    year = {2014}
}
        """
    ),
    _Publication(
        "Robust Dialog State Tracking Using Delexicalised Recurrent Neural Networks and Unsupervised Adaptation",
        "Matthew Henderson, Blaise Thomson, and Steve Young",
        "2014",
        """
        This paper presents a method to robustly transfer recurrent neural networks from one
        dialog domain to another, including a technique for adapting the parameters online with
        no supervision. This was the top performing tracker in the Third Dialog State Tracking
        Challenge. Presented as a poster at IEEE SLT 2014, in Lake Tahoe.
        """,
        "assets/pdf/Robust_Dialog_State_Tracking_Using_Delexicalised_Recurrent_Neural_Networks_and_Unsupervised_Adaptation.pdf",
        """
@inproceedings{Henderson2014d,
    author = {Henderson, M. and  Thomson, B. and Young, S. J.},
    booktitle = {Proceedings of IEEE Spoken Language Technology},
    title = {{Robust Dialog State Tracking Using Delexicalised Recurrent Neural Networks and Unsupervised Adaptation}},
    year = {2014}
}
        """
    ),
    _Publication(
        "The Second Dialog State Tracking Challenge",
        "Matthew Henderson, Blaise Thomson, and Jason Williams",
        "2014",
        """
        This paper presents the results of the Second Dialog State Tracking Challenge,
        a research challenge focused on improving the state of the art in tracking
        the state of spoken dialog systems. In total 9 research teams from across the
        world competed, with 31 total entries.
        """,
        "assets/pdf/The_Second_Dialog_State_Tracking_Challenge.pdf",
        """
@inproceedings{Henderson2014a,
    author = {Henderson, M. and  Thomson, B. and Williams, J.},
    booktitle = {Proceedings of SIGdial},
    title = {{The Second Dialog State Tracking Challenge}},
    year = {2014}
}
        """
    ),
    _Publication(
        "Word-based Dialog State Tracking with Recurrent Neural Networks",
        "Matthew Henderson, Blaise Thomson, and Steve Young",
        "2014",
        """
        This paper presents a new method for dialog state tracking that relies directly
        on the words spoken, i.e. the output of a speech recogniser, rather than a semantic
        representation. The method is validated in the Second Dialog State Tracking Challenge,
        and found to be one of the top two tracking methods. Nominated for best paper.
        """,
        "assets/pdf/Word_based_Dialog_State_Tracking_with_Recurrent_Neural_Networks.pdf",
        """
@inproceedings{Henderson2014b,
    author = {Henderson, M. and  Thomson, B. and Young, S. J.},
    booktitle = {Proceedings of SIGdial},
    title = {{Word-based Dialog State Tracking with Recurrent Neural Networks}},
    year = {2014}
}
        """
    ),
    _Publication(
        "Deep Neural Network Approach for the Dialog State Tracking Challenge",
        "Matthew Henderson, Blaise Thomson, and Steve Young",
        "2013",
        """
        Inspired by recent promising results using Deep Neural Networks in speech applications,
        this paper shows how to apply DNNs to Dialog State Tracking. This was presented as a
        poster at SIGdial and was entered in the Dialog State Tracking Challenge.
        """,
        "assets/pdf/Deep_Neural_Network_for_Dialog_State_Tracking.pdf",
        """
@inproceedings{Henderson2013a,
    author = { Henderson, M. and  Thomson, B. and Young, S. J.},
    booktitle = {Proceedings of SIGdial},
    title = {{Deep Neural Network Approach for the Dialog State Tracking Challenge}},
    year = {2013}
}
        """
    ),
    _Publication(
        "Discriminative Spoken Language Understanding Using Word Confusion Networks",
        "Matthew Henderson, Milica Gašić, Blaise Thomson, Pirros Tsiakoulis, Kai Yu, and Steve Young",
        "2012",
        """
        This paper presents a new form of semantic decoding which uses the entire
        distribution of speech recognition hypotheses to infer the meaning of an utterance.
        This was presented as a poster at SLT. The dataset used for training the decoder and for
        the offline evaluation is freely available:
        <a href="https://www.repository.cam.ac.uk/handle/1810/248271/" target="_blank">In Car SLU Corpus</a>.
        """,
        "assets/pdf/Discriminative_Spoken_Language_Understanding_Using_Word_Confusion_Networks.pdf",
        """
@inproceedings{Henderson2012a,
    author = {Henderson, Matthew and Ga\v{s}i\'{c}, Milica and Thomson, Blaise and Tsiakoulis, Pirros and Yu, Kai and Young, Steve},
    booktitle = {Spoken Language Technology Workshop, 2012. IEEE},
    title = {{Discriminative Spoken Language Understanding Using Word Confusion Networks}},
    year = {2012}
}
        """
    ),
    _Publication(
        "Recovering from Non-Understanding Errors in a Conversational Dialogue System",
        "Matthew Henderson, Colin Matheson, and Jon Oberlander",
        "2012",
        """
        A paper stemming from work done in my MSc evaluating a set of strategies a conversational
        (rather than task-oriented) dialogue system can use to recover from situations where
        it is unable to understand what the user has asked. The dialogue system concerned is
        for a robot tourguide which shows visitors around the Edinburgh Informatics Forum.
        This was an oral presentation at Semdial.
        """,
        "assets/pdf/Recovering_From_Non_Understanding_Errors_in_a_Conversational_Dialogue_System.pdf",
        """
@inproceedings{Henderson2012a,
    author = {Henderson, Matthew and Matheson, Colin and Oberlander, Jon},
    booktitle = {Workshop on the Semantics and Pragmatics of Dialogue},
    title = {{Recovering from Non-Understanding Errors in a Conversational Dialogue System}},
    year = {2012}
}
        """
    )
]
