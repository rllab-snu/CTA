from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.cta import CTAAgent
from agents.dualhiql import DualHIQLAgent
from agents.dualgcivl import DualGCIVLAgent
from agents.gcbcta import GCBCTAAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cta=CTAAgent,
    dualhiql=DualHIQLAgent,
    dualgcivl=DualGCIVLAgent,
    gcbcta=GCBCTAAgent,
)
