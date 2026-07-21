# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitModel

AutoConfig.register(HGRNBitConfig.model_type, HGRNBitConfig)
AutoModel.register(HGRNBitConfig, HGRNBitModel)
AutoModelForCausalLM.register(HGRNBitConfig, HGRNBitForCausalLM)


__all__ = ['HGRNBitConfig', 'HGRNBitForCausalLM', 'HGRNBitModel']
