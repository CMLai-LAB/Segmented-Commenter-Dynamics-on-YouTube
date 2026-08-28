#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

TRADITIONAL_MEDIA_CHANNELS = {
    "中天新聞",
    "壹電視NEXT TV",
    "年代新聞CH50",
    "東森新聞 CH51",
    "TVBS NEWS",
    "三立新聞網SETN",
    "民視新聞網 Formosa TV News network",
    "中視新聞",
    "台視新聞 TTV NEWS",
    "華視新聞 CH52",
    "公視新聞網",
}

EMERGING_MEDIA_CHANNELS = {
    "風傳媒 The Storm Media",
    "CNEWS匯流新聞網",
    "ETtoday新聞雲",
    "NOWNEWS",
    "鏡週刊",
    "鏡新聞",
    "品觀點",
    "Yahoo風向",
    "Yahoo TV 一起看",
    "卡提諾狂新聞",
    "老天鵝娛樂",
    "CTWANT",
}

MEDIA_TYPE_LABELS = {
    "traditional": "Traditional Media",
    "emerging": "Digital / Emerging Media",
    "unknown": "Unknown",
}

CHANNEL_DISPLAY_LABELS = {
    "中天新聞": "CTI News",
    "壹電視NEXT TV": "Next TV",
    "年代新聞CH50": "ERA News",
    "東森新聞 CH51": "EBC News",
    "TVBS NEWS": "TVBS News",
    "三立新聞網SETN": "SET News",
    "民視新聞網 Formosa TV News network": "FTV News",
    "中視新聞": "China TV",
    "台視新聞 TTV NEWS": "TTV News",
    "華視新聞 CH52": "CTS News",
    "公視新聞網": "PTS News",
    "風傳媒 The Storm Media": "The Storm Media",
    "CNEWS匯流新聞網": "CNEWS",
    "ETtoday新聞雲": "ETtoday News",
    "NOWNEWS": "NOW News",
    "鏡週刊": "Mirror Weekly",
    "鏡新聞": "Mirror News",
    "品觀點": "Pin View Media",
    "Yahoo風向": "Yahoo Trends",
    "Yahoo TV 一起看": "Yahoo TV",
    "卡提諾狂新聞": "Crazy News",
    "老天鵝娛樂": "OMGooseTW",
    "CTWANT": "CTWANT",
}


def media_type_for_channel(channel: str | None) -> str:
    channel = (channel or "").strip()
    if channel in TRADITIONAL_MEDIA_CHANNELS:
        return "traditional"
    if channel in EMERGING_MEDIA_CHANNELS:
        return "emerging"
    return "unknown"


def display_label_for_channel(channel: str | None) -> str:
    channel = (channel or "").strip()
    return CHANNEL_DISPLAY_LABELS.get(channel, channel)
