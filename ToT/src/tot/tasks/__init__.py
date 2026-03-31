def get_task(name,step):
    if name == 'game24':
        from ..tasks.game24 import Game24Task
        return Game24Task(step)
    elif name == 'text':
        from ..tasks.text import TextTask
        return TextTask(step)
    elif name == 'crosswords':
        from ..tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask(step)
    elif name == 'winograd':
        from ..tasks.winograd import PronounDisambiguationTask
        return PronounDisambiguationTask(step)
    elif name == 'logibench_bqa':
        from ..tasks.logibench_bqa import Logibench_BQA
        return Logibench_BQA(step)
    elif name == 'logibench_mcqa':
        from ..tasks.logibench_mcqa import Logibench_MCQA
        return Logibench_MCQA(step)
    elif name == 'commonsenseqa':
        from ..tasks.commonsenseqa import Commonsenseqa
        return Commonsenseqa(step)
    elif name == 'cosmosqa':
        from ..tasks.cosmosqa import Cosmos
        return Cosmos(step)
    elif name == 'medqa':
        from ..tasks.medqa import MedQaTask
        return MedQaTask(step)
    elif name == 'sciq':
        from ..tasks.sciq import SciqTask
        return SciqTask(step)
    elif name == '2WikiMultiHopQA':
        from ..tasks.WikiMultiHopQA import WikiMultiHopQATask
        return WikiMultiHopQATask(step)
    elif name == 'squad':
        from ..tasks.squad import SquadTask
        return SquadTask(step)
    elif name == 'hotpotqa':
        from ..tasks.hotpotqa import HotpotqaTask 
        return HotpotqaTask(step)
    else:
        raise NotImplementedError