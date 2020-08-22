#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:12:32 2020

@author: winston
"""
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM, TimeDistributed, Bidirectional, Input




def FixedConcat_SE(feat_inp_spec, feat_out_spec, trained_prosodic_model):
    inputs = Input((None, feat_inp_spec))
    # BLSTM Structure
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(inputs)
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(encode)
    encode = TimeDistributed(Dense(300))(encode)
    encode = LeakyReLU()(encode)
    decode = TimeDistributed(Dense(feat_out_spec))(encode)
    enhance_spec = LeakyReLU()(decode)
    # pre-trained and fixed FE model to reconstruct prosodic features
    recon_prosodic = trained_prosodic_model(enhance_spec)
    # final graph
    model = Model(inputs=inputs, outputs=[enhance_spec, recon_prosodic])
    return model


def JointConcat_SE(feat_inp_spec, feat_out_spec, feat_out_prosodic):
    inputs = Input((None, feat_inp_spec))
    # LSTM SE Part
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(inputs)
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(encode)
    encode = TimeDistributed(Dense(300))(encode)
    encode = LeakyReLU()(encode)
    decode = TimeDistributed(Dense(feat_out_spec))(encode)
    enhance_spec = LeakyReLU()(decode)
    # FE Part
    fe_decode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(enhance_spec)
    fe_decode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(fe_decode)
    fe_decode = TimeDistributed(Dense(300))(fe_decode)
    fe_decode = LeakyReLU()(fe_decode)
    fe_decode = TimeDistributed(Dense(feat_out_prosodic))(fe_decode)
    enhance_prosodic = LeakyReLU()(fe_decode)
    # final graph
    model = Model(inputs=inputs, outputs=[enhance_spec, enhance_prosodic])
    return model


def MultiTask_SE(feat_inp_spec, feat_out_spec, feat_out_prosodic):
    inputs = Input((None, feat_inp_spec))
    # BLSTM Structure (Shared-Layer)
    shared = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(inputs)
    shared = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(shared)
    shared = TimeDistributed(Dense(300))(shared)
    shared = LeakyReLU()(shared)
    # Task1: Spec
    decode1 = TimeDistributed(Dense(feat_out_spec))(shared)
    enhance_spec = LeakyReLU()(decode1)
    # Task2: Praat
    decode2 = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(shared)
    decode2 = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(decode2)
    decode2 = TimeDistributed(Dense(300))(decode2)
    decode2 = LeakyReLU()(decode2)
    decode2 = TimeDistributed(Dense(feat_out_prosodic))(decode2)
    enhance_prosodic = LeakyReLU()(decode2)
    # final graph
    model = Model(inputs=inputs, outputs=[enhance_spec, enhance_prosodic])
    return model


def FE_model(feat_inp_spec, feat_out_prosodic):
    inputs = Input((None, feat_inp_spec))
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(inputs)
    encode = Bidirectional(LSTM(feat_inp_spec, return_sequences=True), merge_mode='concat')(encode)
    encode = TimeDistributed(Dense(300))(encode)
    encode = LeakyReLU()(encode)
    decode = TimeDistributed(Dense(feat_out_prosodic))(encode)
    decode_prosodic = LeakyReLU()(decode)
    # final graph
    model = Model(inputs=inputs, outputs=decode_prosodic)
    return model


