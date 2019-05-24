#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 6 15:01:32 2019

@author: titian
"""
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.inf)

# market parameters
total_supply = 1000000
max_price = 1000
min_mint = 10

# curve parameters
a = max_price/2
b = total_supply/2
c = total_supply*1000

k = 500 # vertical
h = 100000 # horizontal

# supply vector
x = np.arange(0., total_supply + min_mint, min_mint)

# general sigmoidal TBC 
def price(x, a, b, c):
    return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

def collateral(x, a, b, c):
    return a * (np.sqrt(b**2 - 2 * b * x + c + x**2) + x) - (a*np.sqrt(b**2 + c))

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 6))

# price graph
ax11.plot(x, price(x, a, b, c))
ax11.fill_between(x, price(x, a, b, c), alpha=0.5)
ax11.set_xlabel("Supply", fontweight='bold')
ax11.set_ylabel("Price", fontweight='bold')
ax11.set_xlim([0, np.max(x)])
ax11.set_xticks([0, total_supply/2, total_supply])
ax11.set_xticklabels([0, "b", "2b"])
ax11.set_ylim(bottom=0)
ax11.set_yticks([0, max_price/2, max_price])
ax11.set_yticklabels([0, "a", "2a"])
ax11.grid(axis="y", linestyle="--", alpha=0.7)

# collateral graph
ax12.plot(x, collateral(x, a, b, c), color='#ff7f0e')
ax12.set_xlabel("Supply", fontweight='bold')
ax12.set_ylabel("Collateral", fontweight='bold')
ax12.set_xlim([0, total_supply])
ax12.set_xticks([0, total_supply/2, total_supply])
ax12.set_xticklabels([0, "b", "2b"])
ax12.set_ylim(bottom=0)
ax12.set_yticks([0, 2*total_supply/2 * max_price/2])
ax12.set_yticklabels([0, "2ab"])
ax12.grid(axis="y", linestyle="--", alpha=0.7)

# constant curve taxation
def price_buy_const(x, a, b, c, k):
	return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1) + k

def price_sell_const(x, a, b, c):
	return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

fig2 = plt.figure(figsize=(12, 10))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig2)
ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig2)
ax31 = plt.subplot2grid((2, 2), (1, 1), fig=fig2)
ax32 = ax31.twinx()

# price curves
ax1.plot(x, price_buy_const(x, a, b, c, k), label="Buy")
ax1.plot(x, price_sell_const(x,a, b, c), label="Sell")
ax1.set_xlabel("Supply", fontweight='bold')
ax1.set_ylabel("Price", fontweight='bold')
ax1.set_xlim([0, np.max(x)])
ax1.set_xticks([0, total_supply/2, total_supply])
ax1.set_xticklabels([0, "b", "2b"])
ax1.set_ylim(bottom=0)
ax1.set_yticks([0, k, max_price, max_price + k])
ax1.set_yticklabels([0, "k", "2a", "2a+k"])
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend()

# collateral curves
def coll_buy_const(x, a, b, c, k):
	return a * (np.sqrt(b**2 - 2 * b * x + c + x**2) + x) + (k - a*np.sqrt(b**2 + c)) + k*x

def coll_sell_const(x, a, b, c):
	return a * (np.sqrt(b**2 - 2 * b * x + c + x**2) + x) - (a*np.sqrt(b**2 + c))

line1 = ax2.plot(x, coll_buy_const(x, a, b, c, k), label="Buy", color='#1f77b4')
line2 = ax2.plot(x, coll_sell_const(x, a, b, c), label="Sell", color='#ff7f0e')
ax2.set_xlabel("Supply", fontweight='bold')
ax2.set_ylabel("Collateral", fontweight='bold')
ax2.set_xlim([0, np.max(x)])
ax2.set_xticks([0, total_supply/2, total_supply])
ax2.set_xticklabels([0, "b", "2b"])
ax2.set_ylim(bottom=0)
ax2.set_yticks([0, 2*max_price/2*total_supply/2, max_price*total_supply + k])
ax2.set_yticklabels([0, "2ab", "4ab+k"])
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend()

# tax rate & amount
def tax_rate_const():
	return (price_buy_const(x, a, b, c, k) - price_sell_const(x, a, b, c))/price_buy_const(x, a, b, c, k)

def tax_amount_const():
	return price_buy_const(x, a, b, c, k) - price_sell_const(x, a, b, c)

line1 = ax31.plot(x, tax_rate_const(), label="Tax Rate", color="#2ca02c")
line2 = ax32.plot(x, tax_amount_const(), label="Tax Amount", color="#d62728")
ax31.set_xlabel("Supply", fontweight='bold')
ax31.set_ylabel("Rate", fontweight='bold', color='#2ca02c', labelpad=-10)
ax31.set_xlim([0, np.max(x)])
ax31.set_xticks([0, total_supply/2, total_supply])
ax31.set_xticklabels([0, "b", "2b"])

ax31.set_ylim(bottom=0)
ax31.set_yticks([0.0, k/(max_price+k), 1.0])
ax31.set_yticklabels([0.0, r'$\frac{k}{2a+k}$', 1.0], color="#2ca02c")
ax31.grid(axis="y", linestyle="--", alpha=0.7, color="#2ca02c")

ax32.set_ylabel("Amount", fontweight='bold', color="#d62728")
ax32.set_yticks([k])
ax32.set_yticklabels(["k"], color="#d62728")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax31.legend(lines, labels)

# bell curve taxation
def price_buy_bell(x, a, b, c):
	return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

def price_sell_bell(x, a, b, c, h):
	return a * ((x - h - b) / np.sqrt(c + (x - h - b)**2) + 1)

fig3 = plt.figure(figsize=(12, 10))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig3)
ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig3)
ax31 = plt.subplot2grid((2, 2), (1, 1), fig=fig3)
ax32 = ax31.twinx()

# price curves
ax1.plot(x, price_buy_bell(x, a, b, c), label="Buy", color='#1f77b4')
ax1.plot(x, price_sell_bell(x, a, b, c, h), label="Sell", color='#ff7f0e')
ax1.set_xlabel("Supply", fontweight='bold')
ax1.set_ylabel("Price", fontweight='bold')
ax1.set_xlim([0, np.max(x)])
ax1.set_xticks([0, total_supply/2, total_supply/2 + h, total_supply])
ax1.set_xticklabels([0, "b", "b+h", "2b"])
ax1.set_ylim(bottom=0)
ax1.set_yticks([0, max_price])
ax1.set_yticklabels([0, "2a"])
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend()

# collateral curves
def coll_buy_bell(x, a, b, c):
	return a * (np.sqrt(b**2 - 2 * b * x + c + x**2) + x) - a*np.sqrt(b**2 + c)

def coll_sell_bell(x, a, b, c, h):
	return a * (np.sqrt((b + h - x)**2 + c) + x) - (a*np.sqrt((b + h)**2 + c))

line1 = ax2.plot(x, coll_buy_bell(x, a, b, c), label="Buy", color='#1f77b4')
line2 = ax2.plot(x, coll_sell_bell(x, a, b, c, h), label="Sell", color='#ff7f0e')
ax2.set_xlabel("Supply", fontweight='bold')
ax2.set_ylabel("Collateral", fontweight='bold')
ax2.set_xlim([0, np.max(x)])
ax2.set_xticks([0, total_supply/2, total_supply/2 + h, total_supply])
ax2.set_xticklabels([0, "b", "b+h", "2b"])
ax2.set_ylim(bottom=0)
ax2.set_yticks([0, 2*max_price/2*total_supply/2])
ax2.set_yticklabels([0, "2ab"])
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend()

# tax rate & amount
def tax_rate_bell():
	return (price_buy_bell(x, a, b, c) - price_sell_bell(x, a, b, c, h))/price_buy_bell(x, a, b, c)

def tax_amount_bell():
	return price_buy_bell(x, a, b, c) - price_sell_bell(x, a, b, c, h)

line1 = ax31.plot(x, tax_rate_bell(), label="Tax Rate", color="#2ca02c")
line2 = ax32.plot(x, tax_amount_bell(), label="Tax Amount", color="#d62728")
ax31.set_xlabel("Supply", fontweight='bold')
ax31.set_ylabel("Rate", fontweight='bold', color='#2ca02c', labelpad=-10)
ax31.set_xlim([0, np.max(x)])
ax31.set_xticks([0, total_supply/2, total_supply/2 + h, total_supply])
ax31.set_xticklabels([0, "b", "b+h", "2b"])

ax31.set_ylim([0, 1.0])
ax31.set_yticks([0.0, 1.0])
ax31.set_yticklabels([0.0, 1.0], color="#2ca02c")

ax32.set_ylim(bottom=0)
ax32.set_ylabel("Amount", fontweight='bold', color="#d62728")
ax32.set_yticks([0])
ax32.set_yticklabels([0], color="#d62728")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax31.legend(lines, labels)

# decreasing curve taxation
fig4 = plt.figure(figsize=(12, 10))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig4)
ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig4)
ax31 = plt.subplot2grid((2, 2), (1, 1), fig=fig4)
ax32 = ax31.twinx()

# price curves
def price_buy_dec(x, a, b, c, k):
	return (a - k/2) * ((x - b) / np.sqrt(c + (x - b)**2) + 1) + k

def price_sell_dec(x, a, b, c):
	return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

ax1.plot(x, price_buy_dec(x, a, b, c, k), label="Buy", color='#1f77b4')
ax1.plot(x, price_sell_dec(x, a, b, c), label="Sell", color='#ff7f0e')
ax1.set_xlabel("Supply", fontweight='bold')
ax1.set_ylabel("Price", fontweight='bold')
ax1.set_xlim([0, np.max(x)])
ax1.set_xticks([0, total_supply/2, total_supply])
ax1.set_xticklabels([0, "b", "2b"])
ax1.set_ylim(bottom=0)
ax1.set_yticks([0, k, max_price])
ax1.set_yticklabels([0, "k", "2a"])
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend()

# collateral curves
def coll_buy_dec(x, a, b, c, k):
	return (a - k/2)*(np.sqrt(b**2 - 2 * b * x + c + x**2) + x) + (k - (a - k/2)*np.sqrt(b**2 + c)) + k*x

def coll_sell_dec(x, a, b, c):
	return a*(np.sqrt(b**2 - 2 * b * x + c + x**2) + x) - a*np.sqrt(b**2 + c)

line1 = ax2.plot(x, coll_buy_dec(x, a, b, c, k), label="Buy", color='#1f77b4')
line2 = ax2.plot(x, coll_sell_dec(x, a, b, c), label="Sell", color='#ff7f0e')
ax2.set_xlabel("Supply", fontweight='bold')
ax2.set_ylabel("Collateral", fontweight='bold')
ax2.set_xlim([0, np.max(x)])
ax2.set_xticks([0, total_supply/2, total_supply])
ax2.set_xticklabels([0, "b", "2b"])
ax2.set_ylim(bottom=0)
ax2.set_yticks([0, 2*max_price/2*total_supply/2, 3*a*b + k])
ax2.set_yticklabels([0, "2ab", "3ab+k"])
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend()

# tax rate & amount
def tax_rate_dec():
	return (price_buy_dec(x, a, b, c, k) - price_sell_dec(x, a, b, c))/price_buy_dec(x, a, b, c, k)

def tax_amount_dec():
	return price_buy_dec(x, a, b, c, k) - price_sell_dec(x, a, b, c)

line1 = ax31.plot(x, tax_rate_dec(), label="Tax Rate", color="#2ca02c")
line2 = ax32.plot(x, tax_amount_dec(), label="Tax Amount", color="#d62728")
ax31.set_xlabel("Supply", fontweight='bold')
ax31.set_xlim([0, np.max(x)])
ax31.set_xticks([0, total_supply/2, total_supply])
ax31.set_xticklabels([0, "b", "2b"])

ax31.set_ylabel("Rate", fontweight='bold', color='#2ca02c', labelpad=-10)
ax31.set_ylim([0, 1.0])
ax31.set_yticks([0.0, 1.0])
ax31.set_yticklabels([0.0, 1.0], color="#2ca02c")

ax32.set_ylabel("Amount", fontweight='bold', color="#d62728")
ax32.set_ylim(bottom=0)
ax32.set_yticks([0, k])
ax32.set_yticklabels([0, "k"], color="#d62728")
ax32.grid(axis="y", linestyle="--", alpha=0.7, color="#d62728")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax32.legend(lines, labels)

# increasing curve taxation
fig5 = plt.figure(figsize=(12, 10))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig5)
ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig5)
ax31 = plt.subplot2grid((2, 2), (1, 1), fig=fig5)
ax32 = ax31.twinx()

# price curves
def price_buy_inc(x, a, b, c, k):
	return (a + k/2) * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

def price_sell_inc(x, a, b, c):
	return a * ((x - b) / np.sqrt(c + (x - b)**2) + 1)

ax1.plot(x, price_buy_inc(x, a, b, c, k), label="Buy", color='#1f77b4')
ax1.plot(x, price_sell_inc(x, a, b, c), label="Sell", color='#ff7f0e')
ax1.set_xlabel("Supply", fontweight='bold')
ax1.set_ylabel("Price", fontweight='bold')
ax1.set_xlim([0, np.max(x)])
ax1.set_xticks([0, total_supply/2, total_supply])
ax1.set_xticklabels([0, "b", "2b"])
ax1.set_ylim(bottom=0)
ax1.set_yticks([0, max_price, max_price + k])
ax1.set_yticklabels([0, "2a", "2a+k"])
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.legend()

# collateral curves
def coll_buy_inc(x, a, b, c, k):
	return (a + k/2) * (np.sqrt(b**2 - 2*b*x + c + x**2) + x) - (a + k/2)*np.sqrt(b**2 + c)

def coll_sell_inc(x, a, b, c):
	return a * (np.sqrt(b**2 - 2*b*x + c + x**2) + x) - a*np.sqrt(b**2 + c)

line1 = ax2.plot(x, coll_buy_inc(x, a, b, c, k), label="Buy", color='#1f77b4')
line2 = ax2.plot(x, coll_sell_inc(x, a, b, c), label="Sell", color='#ff7f0e')
ax2.set_xlabel("Supply", fontweight='bold')
ax2.set_ylabel("Collateral", fontweight='bold')
ax2.set_xlim([0, np.max(x)])
ax2.set_xticks([0, total_supply/2, total_supply])
ax2.set_xticklabels([0, "b", "2b"])
ax2.set_ylim(bottom=0)
ax2.set_yticks([0, 2*max_price/2*total_supply/2, 3*a*b + k])
ax2.set_yticklabels([0, "2ab", "3ab+k"])
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.legend()

# tax rate & amount
def tax_rate_inc():
	return (price_buy_inc(x, a, b, c, k) - price_sell_inc(x, a, b, c))/price_buy_inc(x, a, b, c, k)

def tax_amount_inc():
	return price_buy_inc(x, a, b, c, k) - price_sell_inc(x, a, b, c)

line1 = ax31.plot(x, tax_rate_inc(), label="Tax Rate", color="#2ca02c")
line2 = ax32.plot(x, tax_amount_inc(), label="Tax Amount", color="#d62728")
ax31.set_xlabel("Supply", fontweight='bold')
ax31.set_xlim([0, np.max(x)])
ax31.set_xticks([0, total_supply/2, total_supply])
ax31.set_xticklabels([0, "b", "2b"])

ax31.set_ylabel("Rate", fontweight='bold', color='#2ca02c', labelpad=-10)
ax31.set_ylim([0, 1.0])
ax31.set_yticks([0.0, k/(2*a+k), 1.0])
ax31.set_yticklabels([0.0, "t", 1.0], color="#2ca02c")

ax32.set_ylabel("Amount", fontweight='bold', color="#d62728")
ax32.set_ylim(bottom=0)
ax32.set_yticks([0, k])
ax32.set_yticklabels([0, "k"], color="#d62728")
ax32.grid(axis="y", linestyle="--", alpha=0.7, color="#d62728")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax32.legend(lines, labels)

# display and save plot
plt.show()
fig1.savefig('sigmoid_tbc.png')
fig2.savefig('const_tax.png')
fig3.savefig('bell_tax.png')
fig4.savefig('dec_tax.png')
fig5.savefig('inc_tax.png')