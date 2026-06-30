# HC Model Results

Generated: 2026-06-04T18:53:40.791742+00:00

## Folds

| name                     | test_start                | test_end                  | purpose         | reason                                                               | btc_return_pct | btc_range_pct |
| ------------------------ | ------------------------- | ------------------------- | --------------- | -------------------------------------------------------------------- | -------------- | ------------- |
| fold1_primary_last7d     | 2026-05-19T20:05:00+00:00 | 2026-05-26T20:05:00+00:00 | live-like check | latest available 7-day window in the built dataset                   |                |               |
| fold2_down_red_week      | 2026-01-30T00:00:00+00:00 | 2026-02-06T00:00:00+00:00 | regime stress   | lowest BTC 7-day close-to-close return before the primary fold       | -25.2794       | 35.3791       |
| fold3_sideways_bull_week | 2025-12-25T00:00:00+00:00 | 2026-01-01T00:00:00+00:00 | regime stress   | non-negative BTC week closest to sideways, tie-broken by lower range | 0.0765         | 4.0806        |

## Table A - By Horizon

| fold                     | side | horizon | n     | base_rate | auc    | signals_070 | precision_070 |
| ------------------------ | ---- | ------- | ----- | --------- | ------ | ----------- | ------------- |
| fold1_primary_last7d     | UP   | 5       | 18144 | 0.2584    | 0.9361 | 3297        | 0.8948        |
| fold1_primary_last7d     | UP   | 15      | 18144 | 0.1449    | 0.9283 | 1553        | 0.7901        |
| fold1_primary_last7d     | UP   | 30      | 18130 | 0.0675    | 0.9371 | 456         | 0.8618        |
| fold1_primary_last7d     | UP   | 60      | 18076 | 0.0710    | 0.8468 | 297         | 0.9360        |
| fold1_primary_last7d     | UP   | 120     | 18076 | 0.1125    | 0.7855 | 270         | 0.7519        |
| fold1_primary_last7d     | UP   | 180     | 18008 | 0.0865    | 0.7940 | 270         | 0.5000        |
| fold1_primary_last7d     | DOWN | 5       | 18144 | 0.2486    | 0.9545 | 3823        | 0.8519        |
| fold1_primary_last7d     | DOWN | 15      | 18144 | 0.1700    | 0.9336 | 2106        | 0.7726        |
| fold1_primary_last7d     | DOWN | 30      | 18130 | 0.0844    | 0.9436 | 804         | 0.8035        |
| fold1_primary_last7d     | DOWN | 60      | 18076 | 0.0976    | 0.7364 | 372         | 0.7446        |
| fold1_primary_last7d     | DOWN | 120     | 18076 | 0.1025    | 0.8021 | 347         | 0.6945        |
| fold1_primary_last7d     | DOWN | 180     | 18008 | 0.1010    | 0.6941 | 293         | 0.6246        |
| fold2_down_red_week      | UP   | 5       | 18228 | 0.3755    | 0.9426 | 6384        | 0.8463        |
| fold2_down_red_week      | UP   | 15      | 18228 | 0.2665    | 0.9200 | 4085        | 0.8029        |
| fold2_down_red_week      | UP   | 30      | 18228 | 0.1751    | 0.8870 | 1481        | 0.8042        |
| fold2_down_red_week      | UP   | 60      | 18228 | 0.1628    | 0.7341 | 964         | 0.6483        |
| fold2_down_red_week      | UP   | 120     | 18228 | 0.1272    | 0.7575 | 958         | 0.5689        |
| fold2_down_red_week      | UP   | 180     | 18228 | 0.1353    | 0.7889 | 945         | 0.7079        |
| fold2_down_red_week      | DOWN | 5       | 18228 | 0.2637    | 0.9303 | 4231        | 0.7814        |
| fold2_down_red_week      | DOWN | 15      | 18228 | 0.1906    | 0.9040 | 2316        | 0.7211        |
| fold2_down_red_week      | DOWN | 30      | 18228 | 0.1130    | 0.8603 | 962         | 0.5489        |
| fold2_down_red_week      | DOWN | 60      | 18228 | 0.1673    | 0.6736 | 431         | 0.8956        |
| fold2_down_red_week      | DOWN | 120     | 18228 | 0.2515    | 0.6548 | 531         | 0.6723        |
| fold2_down_red_week      | DOWN | 180     | 18228 | 0.2293    | 0.6686 | 491         | 0.7495        |
| fold3_sideways_bull_week | UP   | 5       | 18228 | 0.2770    | 0.9305 | 3409        | 0.8727        |
| fold3_sideways_bull_week | UP   | 15      | 18228 | 0.1683    | 0.9221 | 2024        | 0.8043        |
| fold3_sideways_bull_week | UP   | 30      | 18228 | 0.0881    | 0.9475 | 738         | 0.8699        |
| fold3_sideways_bull_week | UP   | 60      | 18228 | 0.0952    | 0.8004 | 509         | 0.8978        |
| fold3_sideways_bull_week | UP   | 120     | 18228 | 0.1028    | 0.8190 | 482         | 0.7137        |
| fold3_sideways_bull_week | UP   | 180     | 18228 | 0.0889    | 0.7789 | 438         | 0.6667        |
| fold3_sideways_bull_week | DOWN | 5       | 18228 | 0.2092    | 0.9414 | 2699        | 0.8544        |
| fold3_sideways_bull_week | DOWN | 15      | 18228 | 0.1299    | 0.9128 | 1429        | 0.7915        |
| fold3_sideways_bull_week | DOWN | 30      | 18228 | 0.0707    | 0.9376 | 763         | 0.8296        |
| fold3_sideways_bull_week | DOWN | 60      | 18228 | 0.0769    | 0.8680 | 434         | 0.9194        |
| fold3_sideways_bull_week | DOWN | 120     | 18228 | 0.1090    | 0.8283 | 419         | 0.8377        |
| fold3_sideways_bull_week | DOWN | 180     | 18228 | 0.0941    | 0.7989 | 378         | 0.7963        |

## Table B - Calibration

| fold                     | side | bucket  | n     | realized_win_rate |
| ------------------------ | ---- | ------- | ----- | ----------------- |
| fold1_primary_last7d     | UP   | 0.5-0.6 | 1883  | 0.4854            |
| fold1_primary_last7d     | UP   | 0.6-0.7 | 1500  | 0.5833            |
| fold1_primary_last7d     | UP   | 0.7-0.8 | 1485  | 0.6970            |
| fold1_primary_last7d     | UP   | 0.8-0.9 | 1733  | 0.7998            |
| fold1_primary_last7d     | UP   | 0.9-1.0 | 3676  | 0.9279            |
| fold1_primary_last7d     | DOWN | 0.5-0.6 | 1990  | 0.4337            |
| fold1_primary_last7d     | DOWN | 0.6-0.7 | 1711  | 0.5167            |
| fold1_primary_last7d     | DOWN | 0.7-0.8 | 1744  | 0.6181            |
| fold1_primary_last7d     | DOWN | 0.8-0.9 | 2020  | 0.7446            |
| fold1_primary_last7d     | DOWN | 0.9-1.0 | 4883  | 0.8962            |
| fold2_down_red_week      | UP   | 0.5-0.6 | 2671  | 0.3796            |
| fold2_down_red_week      | UP   | 0.6-0.7 | 2198  | 0.4413            |
| fold2_down_red_week      | UP   | 0.7-0.8 | 2364  | 0.5381            |
| fold2_down_red_week      | UP   | 0.8-0.9 | 2964  | 0.6211            |
| fold2_down_red_week      | UP   | 0.9-1.0 | 11840 | 0.8703            |
| fold2_down_red_week      | DOWN | 0.5-0.6 | 3834  | 0.4113            |
| fold2_down_red_week      | DOWN | 0.6-0.7 | 2696  | 0.4477            |
| fold2_down_red_week      | DOWN | 0.7-0.8 | 2177  | 0.5195            |
| fold2_down_red_week      | DOWN | 0.8-0.9 | 2398  | 0.6518            |
| fold2_down_red_week      | DOWN | 0.9-1.0 | 5427  | 0.8784            |
| fold3_sideways_bull_week | UP   | 0.5-0.6 | 1934  | 0.4659            |
| fold3_sideways_bull_week | UP   | 0.6-0.7 | 1610  | 0.5491            |
| fold3_sideways_bull_week | UP   | 0.7-0.8 | 1650  | 0.6285            |
| fold3_sideways_bull_week | UP   | 0.8-0.9 | 2239  | 0.7642            |
| fold3_sideways_bull_week | UP   | 0.9-1.0 | 4865  | 0.9394            |
| fold3_sideways_bull_week | DOWN | 0.5-0.6 | 1783  | 0.4711            |
| fold3_sideways_bull_week | DOWN | 0.6-0.7 | 1540  | 0.5831            |
| fold3_sideways_bull_week | DOWN | 0.7-0.8 | 1498  | 0.6716            |
| fold3_sideways_bull_week | DOWN | 0.8-0.9 | 1678  | 0.7759            |
| fold3_sideways_bull_week | DOWN | 0.9-1.0 | 3872  | 0.9398            |

## Table C - By Symbol

| fold                     | side | symbol            | n   | precision | avg_prob | avg_signed_ret_pct | edge_vs_base | rank_group |
| ------------------------ | ---- | ----------------- | --- | --------- | -------- | ------------------ | ------------ | ---------- |
| fold1_primary_last7d     | UP   | BCH_USDT_SWAP     | 12  | 1.0000    | 0.9222   | 5.7792             | 0.8929       | top        |
| fold1_primary_last7d     | UP   | QTUM_USDT_SWAP    | 13  | 1.0000    | 0.8821   | 0.9457             | 0.8929       | top        |
| fold1_primary_last7d     | UP   | SHIB_USDT_SWAP    | 15  | 1.0000    | 0.9032   | 2.2469             | 0.8929       | top        |
| fold1_primary_last7d     | UP   | RESOLV_USDT_SWAP  | 42  | 0.9762    | 0.8950   | 2.5011             | 0.8691       | top        |
| fold1_primary_last7d     | UP   | AERO_USDT_SWAP    | 37  | 0.9730    | 0.9026   | 2.1034             | 0.8659       | top        |
| fold1_primary_last7d     | UP   | ZRO_USDT_SWAP     | 33  | 0.9697    | 0.8916   | 2.1958             | 0.8626       | top        |
| fold1_primary_last7d     | UP   | WLFI_USDT_SWAP    | 31  | 0.9677    | 0.8773   | 2.3728             | 0.8607       | top        |
| fold1_primary_last7d     | UP   | POPCAT_USDT_SWAP  | 29  | 0.9655    | 0.8912   | 2.1240             | 0.8585       | top        |
| fold1_primary_last7d     | UP   | LINEA_USDT_SWAP   | 28  | 0.9643    | 0.9025   | 2.8410             | 0.8572       | top        |
| fold1_primary_last7d     | UP   | MOVE_USDT_SWAP    | 22  | 0.9545    | 0.8659   | 1.0970             | 0.8475       | top        |
| fold1_primary_last7d     | UP   | BIGTIME_USDT_SWAP | 22  | 0.9545    | 0.8841   | 1.2401             | 0.8475       | top        |
| fold1_primary_last7d     | UP   | FLOW_USDT_SWAP    | 22  | 0.9545    | 0.8962   | 2.9627             | 0.8475       | top        |
| fold1_primary_last7d     | UP   | NMR_USDT_SWAP     | 40  | 0.9500    | 0.9228   | 2.8134             | 0.8429       | top        |
| fold1_primary_last7d     | UP   | IOTA_USDT_SWAP    | 20  | 0.9500    | 0.8833   | 1.5982             | 0.8429       | top        |
| fold1_primary_last7d     | UP   | PEPE_USDT_SWAP    | 20  | 0.9500    | 0.8947   | 2.2604             | 0.8429       | top        |
| fold1_primary_last7d     | UP   | OKB_USDT_SWAP     | 16  | 0.5625    | 0.9061   | 1.2280             | 0.4554       | bottom     |
| fold1_primary_last7d     | UP   | F_USDT_SWAP       | 16  | 0.6250    | 0.8510   | 0.6218             | 0.5179       | bottom     |
| fold1_primary_last7d     | UP   | HMSTR_USDT_SWAP   | 31  | 0.6774    | 0.9090   | 1.7681             | 0.5704       | bottom     |
| fold1_primary_last7d     | UP   | YB_USDT_SWAP      | 54  | 0.6852    | 0.8941   | 0.9382             | 0.5781       | bottom     |
| fold1_primary_last7d     | UP   | APR_USDT_SWAP     | 45  | 0.6889    | 0.8627   | 1.3717             | 0.5818       | bottom     |
| fold1_primary_last7d     | UP   | RAY_USDT_SWAP     | 20  | 0.7000    | 0.8797   | 1.4345             | 0.5929       | bottom     |
| fold1_primary_last7d     | UP   | TIA_USDT_SWAP     | 40  | 0.7000    | 0.8865   | 1.6184             | 0.5929       | bottom     |
| fold1_primary_last7d     | UP   | TRUST_USDT_SWAP   | 34  | 0.7059    | 0.8740   | 1.9273             | 0.5988       | bottom     |
| fold1_primary_last7d     | UP   | SSV_USDT_SWAP     | 31  | 0.7097    | 0.8789   | 0.8984             | 0.6026       | bottom     |
| fold1_primary_last7d     | UP   | KGEN_USDT_SWAP    | 53  | 0.7170    | 0.8698   | 1.3479             | 0.6099       | bottom     |
| fold1_primary_last7d     | UP   | ASTER_USDT_SWAP   | 33  | 0.7273    | 0.8915   | 1.5247             | 0.6202       | bottom     |
| fold1_primary_last7d     | UP   | DASH_USDT_SWAP    | 59  | 0.7288    | 0.8786   | 1.8289             | 0.6218       | bottom     |
| fold1_primary_last7d     | UP   | SAPIEN_USDT_SWAP  | 64  | 0.7344    | 0.9038   | 1.4732             | 0.6273       | bottom     |
| fold1_primary_last7d     | UP   | ALGO_USDT_SWAP    | 32  | 0.7500    | 0.8735   | 1.3118             | 0.6429       | bottom     |
| fold1_primary_last7d     | UP   | ARKM_USDT_SWAP    | 48  | 0.7500    | 0.8784   | 1.4994             | 0.6429       | bottom     |
| fold1_primary_last7d     | DOWN | 1INCH_USDT_SWAP   | 10  | 1.0000    | 0.8896   | 1.2266             | 0.8838       | top        |
| fold1_primary_last7d     | DOWN | HBAR_USDT_SWAP    | 14  | 1.0000    | 0.8653   | 1.2871             | 0.8838       | top        |
| fold1_primary_last7d     | DOWN | ASTER_USDT_SWAP   | 27  | 1.0000    | 0.9447   | 2.9412             | 0.8838       | top        |
| fold1_primary_last7d     | DOWN | BAND_USDT_SWAP    | 30  | 0.9667    | 0.9100   | 2.4142             | 0.8504       | top        |
| fold1_primary_last7d     | DOWN | ICX_USDT_SWAP     | 29  | 0.9655    | 0.8888   | 1.4207             | 0.8493       | top        |
| fold1_primary_last7d     | DOWN | MEME_USDT_SWAP    | 23  | 0.9565    | 0.9064   | 1.8234             | 0.8403       | top        |
| fold1_primary_last7d     | DOWN | FLOW_USDT_SWAP    | 23  | 0.9565    | 0.8542   | 1.2168             | 0.8403       | top        |
| fold1_primary_last7d     | DOWN | ETH_USDT_SWAP     | 19  | 0.9474    | 0.9496   | 2.0351             | 0.8311       | top        |
| fold1_primary_last7d     | DOWN | CC_USDT_SWAP      | 37  | 0.9459    | 0.9273   | 2.5432             | 0.8297       | top        |
| fold1_primary_last7d     | DOWN | GPS_USDT_SWAP     | 35  | 0.9429    | 0.9043   | 1.5677             | 0.8266       | top        |
| fold1_primary_last7d     | DOWN | ADA_USDT_SWAP     | 17  | 0.9412    | 0.8605   | 1.1036             | 0.8249       | top        |
| fold1_primary_last7d     | DOWN | F_USDT_SWAP       | 33  | 0.9394    | 0.8873   | 1.4344             | 0.8231       | top        |
| fold1_primary_last7d     | DOWN | BAT_USDT_SWAP     | 32  | 0.9375    | 0.8854   | 1.3745             | 0.8213       | top        |
| fold1_primary_last7d     | DOWN | INIT_USDT_SWAP    | 31  | 0.9355    | 0.8682   | 0.9470             | 0.8192       | top        |
| fold1_primary_last7d     | DOWN | BIGTIME_USDT_SWAP | 29  | 0.9310    | 0.8818   | 1.3708             | 0.8148       | top        |
| fold1_primary_last7d     | DOWN | CRV_USDT_SWAP     | 27  | 0.5926    | 0.8958   | 1.1282             | 0.4763       | bottom     |
| fold1_primary_last7d     | DOWN | BERA_USDT_SWAP    | 57  | 0.5965    | 0.8733   | 1.0148             | 0.4802       | bottom     |
| fold1_primary_last7d     | DOWN | CHZ_USDT_SWAP     | 29  | 0.6207    | 0.9025   | 0.9161             | 0.5044       | bottom     |
| fold1_primary_last7d     | DOWN | JUP_USDT_SWAP     | 57  | 0.6667    | 0.8875   | 1.1230             | 0.5504       | bottom     |
| fold1_primary_last7d     | DOWN | ONT_USDT_SWAP     | 24  | 0.6667    | 0.8617   | 0.7581             | 0.5504       | bottom     |
| fold1_primary_last7d     | DOWN | LPT_USDT_SWAP     | 52  | 0.6731    | 0.8908   | 1.7681             | 0.5568       | bottom     |
| fold1_primary_last7d     | DOWN | KSM_USDT_SWAP     | 28  | 0.6786    | 0.8706   | 0.7869             | 0.5623       | bottom     |
| fold1_primary_last7d     | DOWN | SAHARA_USDT_SWAP  | 65  | 0.6923    | 0.8930   | 1.0828             | 0.5761       | bottom     |
| fold1_primary_last7d     | DOWN | ZETA_USDT_SWAP    | 36  | 0.6944    | 0.8798   | 1.1522             | 0.5782       | bottom     |
| fold1_primary_last7d     | DOWN | OL_USDT_SWAP      | 43  | 0.6977    | 0.8813   | 1.2849             | 0.5814       | bottom     |
| fold1_primary_last7d     | DOWN | ETHW_USDT_SWAP    | 30  | 0.7000    | 0.8578   | 0.9722             | 0.5838       | bottom     |
| fold1_primary_last7d     | DOWN | XPL_USDT_SWAP     | 67  | 0.7015    | 0.9078   | 1.4359             | 0.5852       | bottom     |
| fold1_primary_last7d     | DOWN | PYTH_USDT_SWAP    | 47  | 0.7021    | 0.8858   | 0.9671             | 0.5859       | bottom     |
| fold1_primary_last7d     | DOWN | OKB_USDT_SWAP     | 34  | 0.7059    | 0.8996   | 1.3737             | 0.5896       | bottom     |
| fold1_primary_last7d     | DOWN | LAYER_USDT_SWAP   | 17  | 0.7059    | 0.8840   | 0.7095             | 0.5896       | bottom     |
| fold2_down_red_week      | UP   | TRX_USDT_SWAP     | 16  | 1.0000    | 0.8928   | 1.8917             | 0.8142       | top        |
| fold2_down_red_week      | UP   | HUMA_USDT_SWAP    | 84  | 0.9286    | 0.9231   | 5.1883             | 0.7427       | top        |
| fold2_down_red_week      | UP   | SOL_USDT_SWAP     | 79  | 0.9241    | 0.9267   | 3.7466             | 0.7382       | top        |
| fold2_down_red_week      | UP   | ETH_USDT_SWAP     | 80  | 0.9000    | 0.9276   | 3.5531             | 0.7142       | top        |
| fold2_down_red_week      | UP   | MUBARAK_USDT_SWAP | 69  | 0.8986    | 0.9082   | 4.3540             | 0.7127       | top        |
| fold2_down_red_week      | UP   | UMA_USDT_SWAP     | 64  | 0.8906    | 0.9420   | 3.3890             | 0.7048       | top        |
| fold2_down_red_week      | UP   | TRUMP_USDT_SWAP   | 58  | 0.8793    | 0.9299   | 3.9450             | 0.6935       | top        |
| fold2_down_red_week      | UP   | OKB_USDT_SWAP     | 82  | 0.8780    | 0.9203   | 3.5110             | 0.6922       | top        |
| fold2_down_red_week      | UP   | ATOM_USDT_SWAP    | 49  | 0.8776    | 0.9330   | 2.6823             | 0.6917       | top        |
| fold2_down_red_week      | UP   | CRO_USDT_SWAP     | 55  | 0.8727    | 0.9273   | 2.7643             | 0.6869       | top        |
| fold2_down_red_week      | UP   | BABY_USDT_SWAP    | 91  | 0.8681    | 0.9176   | 3.8761             | 0.6823       | top        |
| fold2_down_red_week      | UP   | H_USDT_SWAP       | 88  | 0.8636    | 0.9112   | 4.3354             | 0.6778       | top        |
| fold2_down_red_week      | UP   | ETHW_USDT_SWAP    | 72  | 0.8611    | 0.9483   | 3.3915             | 0.6753       | top        |
| fold2_down_red_week      | UP   | SHIB_USDT_SWAP    | 64  | 0.8594    | 0.9297   | 2.9138             | 0.6735       | top        |
| fold2_down_red_week      | UP   | AERO_USDT_SWAP    | 90  | 0.8556    | 0.9205   | 3.5239             | 0.6697       | top        |
| fold2_down_red_week      | UP   | RESOLV_USDT_SWAP  | 54  | 0.5741    | 0.8881   | 1.2527             | 0.3882       | bottom     |
| fold2_down_red_week      | UP   | AEVO_USDT_SWAP    | 58  | 0.6379    | 0.9208   | 2.2321             | 0.4521       | bottom     |
| fold2_down_red_week      | UP   | HOME_USDT_SWAP    | 60  | 0.6500    | 0.8937   | 1.8348             | 0.4642       | bottom     |
| fold2_down_red_week      | UP   | SAPIEN_USDT_SWAP  | 87  | 0.6667    | 0.9259   | 2.5631             | 0.4808       | bottom     |
| fold2_down_red_week      | UP   | MET_USDT_SWAP     | 91  | 0.6703    | 0.8898   | 2.0520             | 0.4845       | bottom     |
| fold2_down_red_week      | UP   | LQTY_USDT_SWAP    | 40  | 0.6750    | 0.8677   | 1.0677             | 0.4892       | bottom     |
| fold2_down_red_week      | UP   | FIL_USDT_SWAP     | 71  | 0.6761    | 0.9194   | 2.8554             | 0.4902       | bottom     |
| fold2_down_red_week      | UP   | AT_USDT_SWAP      | 28  | 0.6786    | 0.8710   | 1.0362             | 0.4927       | bottom     |
| fold2_down_red_week      | UP   | WOO_USDT_SWAP     | 82  | 0.6829    | 0.9240   | 3.0726             | 0.4971       | bottom     |
| fold2_down_red_week      | UP   | PLUME_USDT_SWAP   | 87  | 0.6897    | 0.9267   | 2.6834             | 0.5038       | bottom     |
| fold2_down_red_week      | UP   | TRUST_USDT_SWAP   | 97  | 0.6907    | 0.9099   | 2.8649             | 0.5049       | bottom     |
| fold2_down_red_week      | UP   | CRV_USDT_SWAP     | 78  | 0.6923    | 0.9286   | 2.8823             | 0.5065       | bottom     |
| fold2_down_red_week      | UP   | GPS_USDT_SWAP     | 70  | 0.7000    | 0.8949   | 1.3019             | 0.5142       | bottom     |
| fold2_down_red_week      | UP   | ENJ_USDT_SWAP     | 74  | 0.7027    | 0.9179   | 2.3584             | 0.5169       | bottom     |
| fold2_down_red_week      | UP   | AUCTION_USDT_SWAP | 112 | 0.7054    | 0.8993   | 2.6599             | 0.5195       | bottom     |
| fold2_down_red_week      | DOWN | HOME_USDT_SWAP    | 40  | 0.9000    | 0.9165   | 1.7367             | 0.7163       | top        |
| fold2_down_red_week      | DOWN | ANIME_USDT_SWAP   | 84  | 0.8929    | 0.9082   | 2.9895             | 0.7091       | top        |
| fold2_down_red_week      | DOWN | ENJ_USDT_SWAP     | 37  | 0.8919    | 0.8911   | 1.9859             | 0.7081       | top        |
| fold2_down_red_week      | DOWN | PLUME_USDT_SWAP   | 44  | 0.8864    | 0.9133   | 2.5096             | 0.7026       | top        |
| fold2_down_red_week      | DOWN | TON_USDT_SWAP     | 33  | 0.8788    | 0.8944   | 2.0424             | 0.6950       | top        |
| fold2_down_red_week      | DOWN | AT_USDT_SWAP      | 24  | 0.8750    | 0.8736   | 1.7642             | 0.6913       | top        |
| fold2_down_red_week      | DOWN | LQTY_USDT_SWAP    | 31  | 0.8710    | 0.8943   | 3.2825             | 0.6872       | top        |
| fold2_down_red_week      | DOWN | CVX_USDT_SWAP     | 46  | 0.8696    | 0.9095   | 3.0257             | 0.6858       | top        |
| fold2_down_red_week      | DOWN | H_USDT_SWAP       | 81  | 0.8642    | 0.9082   | 3.3856             | 0.6805       | top        |
| fold2_down_red_week      | DOWN | BAND_USDT_SWAP    | 44  | 0.8636    | 0.9033   | 2.3216             | 0.6799       | top        |
| fold2_down_red_week      | DOWN | HUMA_USDT_SWAP    | 43  | 0.8605    | 0.8969   | 1.4255             | 0.6767       | top        |
| fold2_down_red_week      | DOWN | ACT_USDT_SWAP     | 64  | 0.8594    | 0.9092   | 2.6018             | 0.6756       | top        |
| fold2_down_red_week      | DOWN | LUNA_USDT_SWAP    | 54  | 0.8519    | 0.8854   | 2.2559             | 0.6681       | top        |
| fold2_down_red_week      | DOWN | BARD_USDT_SWAP    | 67  | 0.8507    | 0.8909   | 3.3320             | 0.6670       | top        |
| fold2_down_red_week      | DOWN | TRUMP_USDT_SWAP   | 20  | 0.8500    | 0.8809   | 1.5362             | 0.6663       | top        |
| fold2_down_red_week      | DOWN | SHELL_USDT_SWAP   | 44  | 0.5227    | 0.8548   | 1.1531             | 0.3390       | bottom     |
| fold2_down_red_week      | DOWN | IP_USDT_SWAP      | 76  | 0.5658    | 0.8912   | 1.0312             | 0.3820       | bottom     |
| fold2_down_red_week      | DOWN | OKB_USDT_SWAP     | 28  | 0.5714    | 0.8585   | 1.1306             | 0.3877       | bottom     |
| fold2_down_red_week      | DOWN | RAY_USDT_SWAP     | 60  | 0.6000    | 0.8641   | 1.5397             | 0.4163       | bottom     |
| fold2_down_red_week      | DOWN | YGG_USDT_SWAP     | 43  | 0.6047    | 0.8836   | 0.9386             | 0.4209       | bottom     |
| fold2_down_red_week      | DOWN | KGEN_USDT_SWAP    | 33  | 0.6061    | 0.8939   | 0.6869             | 0.4223       | bottom     |
| fold2_down_red_week      | DOWN | BEAT_USDT_SWAP    | 82  | 0.6098    | 0.8877   | 1.8214             | 0.4260       | bottom     |
| fold2_down_red_week      | DOWN | DASH_USDT_SWAP    | 58  | 0.6207    | 0.8865   | 1.2800             | 0.4369       | bottom     |
| fold2_down_red_week      | DOWN | BNB_USDT_SWAP     | 24  | 0.6250    | 0.8991   | 1.4179             | 0.4413       | bottom     |
| fold2_down_red_week      | DOWN | ETH_USDT_SWAP     | 35  | 0.6286    | 0.8783   | 1.2882             | 0.4448       | bottom     |
| fold2_down_red_week      | DOWN | BICO_USDT_SWAP    | 38  | 0.6316    | 0.8754   | 1.1381             | 0.4478       | bottom     |
| fold2_down_red_week      | DOWN | 0G_USDT_SWAP      | 49  | 0.6327    | 0.8557   | 1.6903             | 0.4489       | bottom     |
| fold2_down_red_week      | DOWN | SOL_USDT_SWAP     | 33  | 0.6364    | 0.8884   | 1.1958             | 0.4526       | bottom     |
| fold2_down_red_week      | DOWN | JUP_USDT_SWAP     | 61  | 0.6393    | 0.8966   | 2.2925             | 0.4556       | bottom     |
| fold2_down_red_week      | DOWN | HMSTR_USDT_SWAP   | 28  | 0.6429    | 0.8668   | 0.8859             | 0.4591       | bottom     |
| fold3_sideways_bull_week | UP   | ENS_USDT_SWAP     | 22  | 1.0000    | 0.9168   | 1.6184             | 0.8799       | top        |
| fold3_sideways_bull_week | UP   | XAU_USDT_SWAP     | 11  | 1.0000    | 0.8981   | 3.6534             | 0.8799       | top        |
| fold3_sideways_bull_week | UP   | OKB_USDT_SWAP     | 11  | 1.0000    | 0.8499   | 1.1142             | 0.8799       | top        |
| fold3_sideways_bull_week | UP   | XRP_USDT_SWAP     | 13  | 1.0000    | 0.9164   | 1.7653             | 0.8799       | top        |
| fold3_sideways_bull_week | UP   | ZRO_USDT_SWAP     | 18  | 1.0000    | 0.9062   | 2.0654             | 0.8799       | top        |
| fold3_sideways_bull_week | UP   | HOME_USDT_SWAP    | 41  | 0.9756    | 0.9129   | 4.0672             | 0.8555       | top        |
| fold3_sideways_bull_week | UP   | LRC_USDT_SWAP     | 37  | 0.9730    | 0.9200   | 2.4959             | 0.8528       | top        |
| fold3_sideways_bull_week | UP   | ZIL_USDT_SWAP     | 32  | 0.9688    | 0.9130   | 1.7605             | 0.8486       | top        |
| fold3_sideways_bull_week | UP   | FIL_USDT_SWAP     | 29  | 0.9655    | 0.9070   | 2.2210             | 0.8454       | top        |
| fold3_sideways_bull_week | UP   | PARTI_USDT_SWAP   | 53  | 0.9623    | 0.9084   | 2.6316             | 0.8421       | top        |
| fold3_sideways_bull_week | UP   | BABY_USDT_SWAP    | 25  | 0.9600    | 0.9001   | 2.1246             | 0.8399       | top        |
| fold3_sideways_bull_week | UP   | ZK_USDT_SWAP      | 50  | 0.9600    | 0.8800   | 2.0485             | 0.8399       | top        |
| fold3_sideways_bull_week | UP   | EGLD_USDT_SWAP    | 47  | 0.9574    | 0.9075   | 2.3915             | 0.8373       | top        |
| fold3_sideways_bull_week | UP   | BICO_USDT_SWAP    | 23  | 0.9565    | 0.8797   | 2.1167             | 0.8364       | top        |
| fold3_sideways_bull_week | UP   | BAT_USDT_SWAP     | 44  | 0.9545    | 0.8891   | 1.8172             | 0.8344       | top        |
| fold3_sideways_bull_week | UP   | OL_USDT_SWAP      | 23  | 0.6522    | 0.8558   | 0.6804             | 0.5320       | bottom     |
| fold3_sideways_bull_week | UP   | H_USDT_SWAP       | 65  | 0.6923    | 0.8746   | 1.1641             | 0.5722       | bottom     |
| fold3_sideways_bull_week | UP   | SEI_USDT_SWAP     | 23  | 0.6957    | 0.8721   | 1.0457             | 0.5755       | bottom     |
| fold3_sideways_bull_week | UP   | ARB_USDT_SWAP     | 27  | 0.7037    | 0.8781   | 1.6402             | 0.5836       | bottom     |
| fold3_sideways_bull_week | UP   | MON_USDT_SWAP     | 98  | 0.7041    | 0.8783   | 1.6431             | 0.5840       | bottom     |
| fold3_sideways_bull_week | UP   | TIA_USDT_SWAP     | 34  | 0.7059    | 0.8741   | 1.4616             | 0.5858       | bottom     |
| fold3_sideways_bull_week | UP   | SATS_USDT_SWAP    | 42  | 0.7143    | 0.8717   | 1.4404             | 0.5942       | bottom     |
| fold3_sideways_bull_week | UP   | PEPE_USDT_SWAP    | 28  | 0.7143    | 0.8904   | 1.5192             | 0.5942       | bottom     |
| fold3_sideways_bull_week | UP   | SHIB_USDT_SWAP    | 32  | 0.7188    | 0.8711   | 1.6561             | 0.5986       | bottom     |
| fold3_sideways_bull_week | UP   | VIRTUAL_USDT_SWAP | 36  | 0.7222    | 0.8738   | 1.4042             | 0.6021       | bottom     |
| fold3_sideways_bull_week | UP   | HMSTR_USDT_SWAP   | 36  | 0.7222    | 0.8691   | 1.4667             | 0.6021       | bottom     |
| fold3_sideways_bull_week | UP   | SAPIEN_USDT_SWAP  | 58  | 0.7241    | 0.8814   | 0.8948             | 0.6040       | bottom     |
| fold3_sideways_bull_week | UP   | PUMP_USDT_SWAP    | 29  | 0.7241    | 0.8804   | 1.8085             | 0.6040       | bottom     |
| fold3_sideways_bull_week | UP   | GAS_USDT_SWAP     | 62  | 0.7258    | 0.8926   | 1.2138             | 0.6057       | bottom     |
| fold3_sideways_bull_week | UP   | ZBT_USDT_SWAP     | 128 | 0.7266    | 0.9161   | 4.5442             | 0.6064       | bottom     |
| fold3_sideways_bull_week | DOWN | BNB_USDT_SWAP     | 11  | 1.0000    | 0.9024   | 1.7274             | 0.8983       | top        |
| fold3_sideways_bull_week | DOWN | MET_USDT_SWAP     | 23  | 1.0000    | 0.8859   | 1.8200             | 0.8983       | top        |
| fold3_sideways_bull_week | DOWN | LTC_USDT_SWAP     | 11  | 1.0000    | 0.9089   | 1.7585             | 0.8983       | top        |
| fold3_sideways_bull_week | DOWN | HUMA_USDT_SWAP    | 15  | 1.0000    | 0.8614   | 2.5515             | 0.8983       | top        |
| fold3_sideways_bull_week | DOWN | POPCAT_USDT_SWAP  | 16  | 1.0000    | 0.8827   | 1.3802             | 0.8983       | top        |
| fold3_sideways_bull_week | DOWN | GLM_USDT_SWAP     | 44  | 0.9773    | 0.8988   | 3.2902             | 0.8756       | top        |
| fold3_sideways_bull_week | DOWN | NMR_USDT_SWAP     | 28  | 0.9643    | 0.8930   | 2.7082             | 0.8626       | top        |
| fold3_sideways_bull_week | DOWN | GRT_USDT_SWAP     | 26  | 0.9615    | 0.8970   | 2.9345             | 0.8599       | top        |
| fold3_sideways_bull_week | DOWN | ZETA_USDT_SWAP    | 26  | 0.9615    | 0.8945   | 2.6526             | 0.8599       | top        |
| fold3_sideways_bull_week | DOWN | RENDER_USDT_SWAP  | 26  | 0.9615    | 0.9064   | 2.4293             | 0.8599       | top        |
| fold3_sideways_bull_week | DOWN | ANIME_USDT_SWAP   | 24  | 0.9583    | 0.8866   | 2.5432             | 0.8567       | top        |
| fold3_sideways_bull_week | DOWN | ZEC_USDT_SWAP     | 45  | 0.9556    | 0.9107   | 4.3753             | 0.8539       | top        |
| fold3_sideways_bull_week | DOWN | CHZ_USDT_SWAP     | 63  | 0.9524    | 0.8892   | 3.6147             | 0.8507       | top        |
| fold3_sideways_bull_week | DOWN | ALGO_USDT_SWAP    | 21  | 0.9524    | 0.9173   | 2.5867             | 0.8507       | top        |
| fold3_sideways_bull_week | DOWN | DASH_USDT_SWAP    | 39  | 0.9487    | 0.9301   | 5.1840             | 0.8471       | top        |
| fold3_sideways_bull_week | DOWN | AAVE_USDT_SWAP    | 11  | 0.5455    | 0.8537   | 0.6531             | 0.4438       | bottom     |
| fold3_sideways_bull_week | DOWN | DOOD_USDT_SWAP    | 22  | 0.5455    | 0.8670   | 0.6871             | 0.4438       | bottom     |
| fold3_sideways_bull_week | DOWN | FLOW_USDT_SWAP    | 94  | 0.6277    | 0.8665   | 1.7041             | 0.5260       | bottom     |
| fold3_sideways_bull_week | DOWN | ATH_USDT_SWAP     | 19  | 0.6316    | 0.8903   | 1.2776             | 0.5299       | bottom     |
| fold3_sideways_bull_week | DOWN | LUNA_USDT_SWAP    | 19  | 0.6316    | 0.8973   | 6.4774             | 0.5299       | bottom     |
| fold3_sideways_bull_week | DOWN | MERL_USDT_SWAP    | 34  | 0.6471    | 0.8728   | 0.9467             | 0.5454       | bottom     |
| fold3_sideways_bull_week | DOWN | S_USDT_SWAP       | 38  | 0.6579    | 0.8907   | 1.3588             | 0.5562       | bottom     |
| fold3_sideways_bull_week | DOWN | BEAT_USDT_SWAP    | 145 | 0.6621    | 0.8651   | 4.6607             | 0.5604       | bottom     |
| fold3_sideways_bull_week | DOWN | CVX_USDT_SWAP     | 12  | 0.6667    | 0.8522   | 0.7166             | 0.5650       | bottom     |
| fold3_sideways_bull_week | DOWN | ZBT_USDT_SWAP     | 128 | 0.6875    | 0.8757   | 3.8488             | 0.5858       | bottom     |
| fold3_sideways_bull_week | DOWN | INIT_USDT_SWAP    | 68  | 0.6912    | 0.8819   | 2.0604             | 0.5895       | bottom     |
| fold3_sideways_bull_week | DOWN | MOODENG_USDT_SWAP | 13  | 0.6923    | 0.8600   | 0.6434             | 0.5906       | bottom     |
| fold3_sideways_bull_week | DOWN | MUBARAK_USDT_SWAP | 26  | 0.6923    | 0.9002   | 3.5800             | 0.5906       | bottom     |
| fold3_sideways_bull_week | DOWN | AT_USDT_SWAP      | 157 | 0.6943    | 0.8757   | 4.4460             | 0.5926       | bottom     |
| fold3_sideways_bull_week | DOWN | ACT_USDT_SWAP     | 82  | 0.6951    | 0.8922   | 1.4394             | 0.5935       | bottom     |

## Table D - Decision Level

| fold                     | action         | count  | win_rate | avg_net_ret_pct |
| ------------------------ | -------------- | ------ | -------- | --------------- |
| fold1_primary_last7d     | LONG           | 6880   | 0.9484   | 1.8325          |
| fold1_primary_last7d     | SHORT          | 8620   | 0.9056   | 1.7665          |
| fold1_primary_last7d     | BOTH_HIGH_SKIP | 0      |          |                 |
| fold1_primary_last7d     | NO_TRADE       | 129215 |          |                 |
| fold1_primary_last7d     | ALL_TRADES     | 15500  | 0.9246   | 1.7958          |
| fold2_down_red_week      | LONG           | 17137  | 0.8713   | 2.9521          |
| fold2_down_red_week      | SHORT          | 9973   | 0.8675   | 1.9355          |
| fold2_down_red_week      | BOTH_HIGH_SKIP | 0      |          |                 |
| fold2_down_red_week      | NO_TRADE       | 118714 |          |                 |
| fold2_down_red_week      | ALL_TRADES     | 27110  | 0.8699   | 2.5781          |
| fold3_sideways_bull_week | LONG           | 8734   | 0.9392   | 2.0953          |
| fold3_sideways_bull_week | SHORT          | 6987   | 0.9181   | 2.6910          |
| fold3_sideways_bull_week | BOTH_HIGH_SKIP | 0      |          |                 |
| fold3_sideways_bull_week | NO_TRADE       | 130103 |          |                 |
| fold3_sideways_bull_week | ALL_TRADES     | 15721  | 0.9298   | 2.3600          |

## Edge Survival

Edge survives folds 2-3: **YES**

CSV: `C:\ml\ml_predictor_v2\docs\HC_MODEL_RESULTS.csv`
