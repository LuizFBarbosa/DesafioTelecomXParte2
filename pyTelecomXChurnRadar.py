"""
╔══════════════════════════════════════════════════════════════════╗
║  TELECOM X — CHURN RADAR                                         ║
║  Identificação de clientes ativos em risco + plano de retenção   ║
║                                                                  ║
║pip install dash dash-bootstrap-components plotly scikit-learn    ║
║                                                                  ║
║  Uso:  python pyTelecomXChurnRadar.py                            ║
║        python pyTelecomXChurnRadar.py meus_dados.csv             ║
║                                                                  ║
║  Luiz Fernando Barbosa                                           ║
╚══════════════════════════════════════════════════════════════════╝
"""
#

INPUT_CSV = "telecomxdados.csv"

import sys, os, warnings, datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ── Paleta ────────────────────────────────────────────────────────
C = dict(
    bg="#080C10", card="#0F1923", card2="#131E2A", border="#1E2D3D",
    text="#E8F4FD", sub="#6B8BA4", blue="#00B4D8", green="#06D6A0",
    red="#EF233C", yellow="#FFB703", purple="#7B2FBE", orange="#FB8500",
    teal="#48CAE4", cyan="#90E0EF",
)
TIER_COLORS = {"Crítico": C['red'], "Alto": C['orange'], "Médio": C['yellow'], "Baixo": C['green']}
TIER_EMOJI  = {"Crítico": "🔴", "Alto": "🟠", "Médio": "🟡", "Baixo": "🟢"}

LAYOUT = dict(
    paper_bgcolor=C["card"], plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="'JetBrains Mono', 'Courier New', monospace", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PIPELINE + PREDIÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_radar_pipeline(csv_path):
    W = 65
    print(f"\n{'═'*W}")
    print(f"  🎯 TELECOM X — CHURN RADAR")
    print(f"     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {csv_path}")
    print(f"{'═'*W}")

    # ── Carga e limpeza ──────────────────────────────────────────
    print("\n📂 [1/6] Carregando dados…")
    df_raw = pd.read_csv(csv_path)
    bool_cols = df_raw.select_dtypes(include='bool').columns
    df_raw[bool_cols] = df_raw[bool_cols].astype(int)
    df_raw['Charges.Total'] = pd.to_numeric(df_raw['Charges.Total'], errors='coerce')
    df_raw['Charges.Total'] = df_raw['Charges.Total'].fillna(df_raw['Charges.Total'].median())
    n_total  = len(df_raw)
    n_churn  = int((df_raw['Churn'] == 1).sum())
    n_active = int((df_raw['Churn'] == 0).sum())
    print(f"   Total: {n_total:,}  |  Histórico churn: {n_churn:,}  |  Ativos: {n_active:,}")

    # Guardar atributos originais dos ativos para enriquecer predição
    mask_active = df_raw['Churn'] == 0
    active_meta = df_raw[mask_active][[
        'tenure', 'Charges.Monthly', 'SeniorCitizen',
        *([c for c in ['Contract', 'Partner', 'Dependents',
                        'PhoneService', 'PaperlessBilling'] if c in df_raw.columns]),
        *([c for c in ['Internet_Fiber optic', 'Internet_DSL'] if c in df_raw.columns]),
    ]].copy()
    active_idx = df_raw[mask_active].index.tolist()

    # ── Preparação do modelo ─────────────────────────────────────
    print("\n⚙  [2/6] Preparando features para modelagem…")
    drop_cols = [c for c in ['customerID','Contract','MultipleLines','OnlineSecurity',
                               'OnlineBackup','DeviceProtection','TechSupport',
                               'StreamingTV','StreamingMovies'] if c in df_raw.columns]
    df_model = df_raw.drop(columns=drop_cols)
    obj_cols = df_model.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df_model = pd.get_dummies(df_model, columns=obj_cols, drop_first=True).astype(int)
    df_model = df_model.apply(lambda c: c.fillna(c.median()) if c.dtype in ['float64','int64'] else c)

    feat_names = [c for c in df_model.columns if c != 'Churn']

    # ── Treinar com dataset histórico completo ───────────────────
    print("\n🤖 [3/6] Treinando modelo ensemble no histórico completo…")
    majority = df_model[df_model['Churn'] == 0]
    minority = df_model[df_model['Churn'] == 1]
    majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    df_bal = pd.concat([majority_down, minority]).sample(frac=1, random_state=42)
    df_bal = df_bal.apply(lambda c: c.fillna(c.median()) if c.dtype in ['float64','int64'] else c)

    X_all = df_bal[feat_names]; y_all = df_bal['Churn']
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                                random_state=42, stratify=y_all)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_te)

    # Ensemble: RF + GBM + LR → média ponderada
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                 random_state=42, class_weight='balanced', n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      random_state=42, subsample=0.8)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    rf.fit(X_tr, y_tr)
    gbm.fit(X_tr, y_tr)
    lr.fit(X_tr_sc, y_te if False else y_tr)

    p_rf  = rf.predict_proba(X_te)[:,1]
    p_gbm = gbm.predict_proba(X_te)[:,1]
    p_lr  = lr.predict_proba(X_te_sc)[:,1]
    p_ens = 0.45*p_rf + 0.35*p_gbm + 0.20*p_lr

    auc_rf  = roc_auc_score(y_te, p_rf)
    auc_gbm = roc_auc_score(y_te, p_gbm)
    auc_ens = roc_auc_score(y_te, p_ens)

    print(f"   Random Forest AUC   : {auc_rf:.4f}")
    print(f"   Gradient Boosting   : {auc_gbm:.4f}")
    print(f"   Ensemble (ponderado): {auc_ens:.4f}  ← usado para predição")

    # Feature importances combinadas
    rf_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

    # ── Predição nos clientes ATIVOS ─────────────────────────────
    print(f"\n🔮 [4/6] Aplicando modelo nos {n_active:,} clientes ativos…")
    X_active = df_model.loc[active_idx, feat_names].copy()
    for col in feat_names:
        if col not in X_active.columns:
            X_active[col] = 0
    X_active = X_active[feat_names]

    X_active_sc = scaler.transform(X_active)
    p_rf_a   = rf.predict_proba(X_active)[:,1]
    p_gbm_a  = gbm.predict_proba(X_active)[:,1]
    p_lr_a   = lr.predict_proba(X_active_sc)[:,1]
    prob_final = 0.45*p_rf_a + 0.35*p_gbm_a + 0.20*p_lr_a

    # ── Montar tabela de resultado ────────────────────────────────
    print("\n📊 [5/6] Segmentando clientes por nível de risco…")
    pred_df = pd.DataFrame({
        'cliente_id':   [f"CLI-{i:05d}" for i in range(len(active_idx))],
        'prob_churn':   np.round(prob_final, 4),
        'risco_pct':    np.round(prob_final * 100, 1),
        'tenure':       active_meta['tenure'].values,
        'monthly':      active_meta['Charges.Monthly'].values,
        'senior':       active_meta['SeniorCitizen'].values,
    })

    if 'Contract' in active_meta.columns:
        pred_df['contrato'] = active_meta['Contract'].values
    if 'PaperlessBilling' in active_meta.columns:
        pred_df['sem_papel'] = active_meta['PaperlessBilling'].values
    if 'Partner' in active_meta.columns:
        pred_df['parceiro'] = active_meta['Partner'].values
    if 'Dependents' in active_meta.columns:
        pred_df['dependentes'] = active_meta['Dependents'].values
    if 'Internet_Fiber optic' in active_meta.columns:
        pred_df['fibra'] = active_meta['Internet_Fiber optic'].values

    pred_df['tier'] = pd.cut(pred_df['prob_churn'],
        bins=[0, 0.30, 0.50, 0.70, 1.01],
        labels=['Baixo', 'Médio', 'Alto', 'Crítico'])

    pred_df['receita_risco'] = pred_df['monthly'] * pred_df['prob_churn']

    # ── Ação recomendada por tier ─────────────────────────────────
    def acao(row):
        if row['prob_churn'] >= 0.70:
            return "🚨 Contato urgente — oferta de retenção personalizada"
        elif row['prob_churn'] >= 0.50:
            return "⚡ Migrar para contrato anual + benefício exclusivo"
        elif row['prob_churn'] >= 0.30:
            return "📞 NPS proativo + programa de fidelidade"
        else:
            return "✅ Monitorar — manter relacionamento ativo"
    pred_df['acao'] = pred_df.apply(acao, axis=1)

    pred_df = pred_df.sort_values('prob_churn', ascending=False).reset_index(drop=True)

    # ── Log de resultados ─────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  RESULTADOS DA PREDIÇÃO — CLIENTES ATIVOS EM RISCO")
    print(f"{'─'*W}")
    tier_summary = pred_df.groupby('tier', observed=True).agg(
        n=('prob_churn','count'),
        prob_media=('prob_churn','mean'),
        ticket_medio=('monthly','mean'),
        tenure_medio=('tenure','mean'),
        receita_risco=('receita_risco','sum'),
    ).sort_index(ascending=False)

    for tier, row in tier_summary.iterrows():
        pct = row['n'] / n_active * 100
        emoji = TIER_EMOJI.get(tier, "")
        print(f"\n  {emoji} TIER {tier.upper()} — {row['n']:,} clientes ({pct:.1f}% da base ativa)")
        print(f"     Probabilidade média de churn : {row['prob_media']*100:.1f}%")
        print(f"     Ticket médio                 : R$ {row['ticket_medio']:.2f}/mês")
        print(f"     Tempo médio como cliente     : {row['tenure_medio']:.0f} meses")
        print(f"     Receita em risco             : R$ {row['receita_risco']:,.0f}/mês")

    total_risco = pred_df['receita_risco'].sum()
    criticos_risco = pred_df[pred_df['tier']=='Crítico']['receita_risco'].sum()
    altos_risco    = pred_df[pred_df['tier']=='Alto']['receita_risco'].sum()

    print(f"\n{'─'*W}")
    print(f"  💰 IMPACTO FINANCEIRO TOTAL")
    print(f"     Receita mensal total em risco (ponderada): R$ {total_risco:,.0f}/mês")
    print(f"     → Tier Crítico + Alto:                    R$ {criticos_risco+altos_risco:,.0f}/mês")
    print(f"     → Se retiver 50% dos críticos:            +R$ {criticos_risco*0.5:,.0f}/mês preservados")
    print(f"     → ROI anual estimado de retenção:         R$ {(criticos_risco+altos_risco*0.5)*0.5*12:,.0f}/ano")

    # Perfil dos críticos
    criticos = pred_df[pred_df['tier']=='Crítico']
    print(f"\n  🔴 PERFIL DETALHADO — TIER CRÍTICO")
    print(f"     Tenure médio  : {criticos['tenure'].mean():.0f} meses (novos!)")
    if 'contrato' in criticos.columns:
        print(f"     Tipo contrato : {criticos['contrato'].value_counts().index[0]}")
    if 'fibra' in criticos.columns:
        print(f"     Fibra ótica   : {criticos['fibra'].mean()*100:.0f}% dos críticos")
    if 'senior' in criticos.columns:
        print(f"     Idosos        : {criticos['senior'].mean()*100:.0f}% dos críticos")

    print(f"\n  🎯 PLANO DE AÇÃO RECOMENDADO:")
    print(f"     Semana 1  → Contato pessoal com os {len(criticos):,} críticos")
    print(f"     Semana 2  → Campanha de ancoragem para os {len(pred_df[pred_df['tier']=='Alto']):,} alto risco")
    print(f"     Mês 1-3   → Onboarding intensivo (tenure < 6 meses)")
    print(f"     Contínuo  → Score semanal automático para monitorar evolução")
    print(f"\n{'═'*W}\n  🌐 Dashboard: http://localhost:8050\n{'═'*W}\n")

    print(f"\n[6/6] Salvando lista de clientes em risco…")
    out_path = csv_path.replace('.csv', '_clientes_em_risco.csv')
    pred_df.to_csv(out_path, index=False)
    print(f"   ✅ {out_path}")

    return dict(
        n_total=n_total, n_active=n_active, n_churn=n_churn,
        pred_df=pred_df,
        tier_summary=tier_summary,
        total_risco=total_risco,
        auc_ens=auc_ens, auc_rf=auc_rf, auc_gbm=auc_gbm,
        rf_imp=rf_imp,
        csv_path=csv_path,
        out_path=out_path,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig_radar_donut(d):
    ts = d['tier_summary']
    order = ['Crítico','Alto','Médio','Baixo']
    vals   = [int(ts.loc[t,'n']) if t in ts.index else 0 for t in order]
    colors = [TIER_COLORS[t] for t in order]
    fig = go.Figure(go.Pie(
        labels=order, values=vals, hole=0.60,
        marker=dict(colors=colors, line=dict(color=C['bg'], width=3)),
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,} clientes<br>%{percent}<extra></extra>",
        direction='clockwise', sort=False,
    ))
    criticos = int(ts.loc['Crítico','n']) if 'Crítico' in ts.index else 0
    altos    = int(ts.loc['Alto','n'])    if 'Alto'    in ts.index else 0
    fig.add_annotation(
        text=f"<b>{criticos+altos:,}</b><br><span style='font-size:11px'>alto risco</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=19, color=C['red']))
    fig.update_layout(**LAYOUT,
        title=dict(text="Mapa de Risco — Base Ativa", font=dict(size=13, color=C['sub'])),
        showlegend=True, legend=dict(font=dict(color=C['text'])))
    return fig

def fig_prob_histogram(pred_df):
    fig = go.Figure()
    bins = np.linspace(0, 1, 30)
    for tier, color in TIER_COLORS.items():
        sub = pred_df[pred_df['tier'] == tier]['prob_churn']
        fig.add_trace(go.Histogram(
            x=sub, name=f"{TIER_EMOJI[tier]} {tier}",
            xbins=dict(start=0, end=1, size=0.035),
            marker_color=color, opacity=0.8,
            hovertemplate=f"<b>{tier}</b><br>P(churn): %{{x:.2f}}<br>Clientes: %{{y}}<extra></extra>",
        ))
    fig.add_vline(x=0.50, line_color=C['yellow'], line_dash='dash', line_width=1.5,
                  annotation_text="threshold 0.5", annotation_font_color=C['yellow'], annotation_font_size=10)
    fig.add_vline(x=0.70, line_color=C['red'], line_dash='dash', line_width=1.5,
                  annotation_text="crítico 0.7", annotation_font_color=C['red'], annotation_font_size=10)
    fig.update_layout(**LAYOUT, barmode='stack',
        title=dict(text="Distribuição de Score de Risco", font=dict(size=13, color=C['sub'])),
        xaxis=dict(title='P(churn)', gridcolor=C['border'], range=[0,1]),
        yaxis=dict(title='Nº de Clientes', gridcolor=C['border']),
        legend=dict(font=dict(color=C['text'])))
    return fig

def fig_scatter_risk(pred_df):
    tier_order = ['Baixo','Médio','Alto','Crítico']
    fig = go.Figure()
    for tier in tier_order:
        sub = pred_df[pred_df['tier'] == tier]
        fig.add_trace(go.Scatter(
            x=sub['tenure'], y=sub['monthly'],
            mode='markers',
            name=f"{TIER_EMOJI[tier]} {tier}",
            marker=dict(
                size=np.clip(sub['prob_churn'] * 18 + 4, 5, 22),
                color=TIER_COLORS[tier], opacity=0.7,
                line=dict(color=C['bg'], width=0.5)),
            hovertemplate=(
                f"<b>{tier}</b><br>"
                "Tenure: %{x} meses<br>"
                "Mensalidade: R$ %{y:.0f}<br>"
                "Score: " + sub['risco_pct'].astype(str) + "%<extra></extra>"
            ),
            customdata=sub[['risco_pct']].values,
        ))
    fig.update_layout(**LAYOUT,
        title=dict(text="Perfil de Risco: Tempo × Mensalidade (tamanho = probabilidade)", font=dict(size=12, color=C['sub'])),
        xaxis=dict(title='Tenure (meses)', gridcolor=C['border']),
        yaxis=dict(title='Mensalidade R$', gridcolor=C['border']),
        legend=dict(font=dict(color=C['text'])),
        height=420)
    return fig

def fig_receita_risco_tier(d):
    ts = d['tier_summary']
    order = ['Crítico','Alto','Médio','Baixo']
    tiers_present = [t for t in order if t in ts.index]
    receita = [float(ts.loc[t,'receita_risco']) for t in tiers_present]
    colors  = [TIER_COLORS[t] for t in tiers_present]
    fig = go.Figure(go.Bar(
        x=tiers_present, y=receita,
        marker_color=colors, opacity=0.85,
        text=[f"R${v/1000:.1f}k" for v in receita],
        textposition='outside', textfont=dict(color=C['text'], size=12),
        hovertemplate="%{x}: R$ %{y:,.0f}/mês<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
        title=dict(text="Receita Mensal em Risco por Tier", font=dict(size=13, color=C['sub'])),
        yaxis=dict(title='R$/mês', gridcolor=C['border']),
        xaxis=dict(gridcolor=C['border']))
    return fig

def fig_tenure_risco(pred_df):
    pred_df2 = pred_df.copy()
    pred_df2['tenure_g'] = pd.cut(pred_df2['tenure'],
        bins=[0,6,12,24,48,72], labels=['0-6m','6-12m','1-2a','2-4a','4+a'])
    grp = pred_df2.groupby('tenure_g', observed=True)['prob_churn'].mean() * 100
    colors = [C['red'] if v >= 50 else C['orange'] if v >= 35 else C['yellow'] if v >= 20 else C['green']
              for v in grp.values]
    fig = go.Figure(go.Bar(
        x=grp.index.astype(str), y=grp.values,
        marker_color=colors,
        text=[f"{v:.0f}%" for v in grp.values],
        textposition='outside', textfont=dict(color=C['text'], size=11),
        hovertemplate="Faixa %{x}: %{y:.1f}% de risco médio<extra></extra>",
    ))
    fig.update_layout(**LAYOUT,
        title=dict(text="Score Médio de Risco por Faixa de Tenure", font=dict(size=13, color=C['sub'])),
        yaxis=dict(title='Score Médio (%)', gridcolor=C['border'], range=[0, max(grp.values)*1.25]),
        xaxis=dict(title='Tempo como cliente', gridcolor=C['border']))
    return fig

def fig_importance(rf_imp):
    top12 = rf_imp.head(12)
    colors = [C['red'] if i < 3 else C['orange'] if i < 6 else C['blue'] for i in range(len(top12))]
    fig = go.Figure(go.Bar(
        x=top12.values[::-1], y=top12.index[::-1],
        orientation='h', marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT, height=420,
        title=dict(text="Variáveis Mais Importantes para Prever Risco", font=dict(size=13, color=C['sub'])),
        xaxis=dict(gridcolor=C['border']),
        yaxis=dict(tickfont=dict(size=10)))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RETENCAO_PLAYBOOK = {
    "Crítico": {
        "titulo": "🚨 Protocolo de Emergência — Crítico",
        "cor": C['red'],
        "acoes": [
            ("Contato pessoal em 48h", "Gerente de conta liga diretamente. Script focado em dor específica do cliente."),
            ("Oferta de downgrade inteligente", "Proponha plano menor antes de perder o cliente. Melhor reduzir ticket que zerar."),
            ("Migração contratual urgente", "Ofereça 20-30% desconto para migrar para contrato anual. ROI positivo a partir do mês 4."),
            ("Diagnóstico técnico gratuito", "Para fibra ótica: agende visita técnica proativa dentro de 72h."),
        ],
        "kpi_meta": "Meta: converter 40% → reduz R$16k+/mês de risco",
    },
    "Alto": {
        "titulo": "⚡ Plano de Intervenção — Alto Risco",
        "cor": C['orange'],
        "acoes": [
            ("Campanha de ancoragem contratual", "Email + SMS oferecendo upgrade para plano anual com 15% desconto + brinde."),
            ("Programa de pontos e benefícios", "Ingresso em programa de fidelidade com cashback de R$10-20/mês nos 6 primeiros meses."),
            ("Pesquisa NPS personalizada", "Envio de NPS com gatilho de resposta em 24h para equipe de CX."),
            ("Bundle de serviços", "Ofereça pacotes adicionais (segurança, backup) com mês gratuito — cria dependência positiva."),
        ],
        "kpi_meta": "Meta: converter 35% → preserva R$19k+/mês",
    },
    "Médio": {
        "titulo": "📞 Engajamento Preventivo — Risco Médio",
        "cor": C['yellow'],
        "acoes": [
            ("Newsletter de valor", "Envio mensal com dicas de uso, novidades e benefícios exclusivos."),
            ("Programa de indicação", "Ofereça desconto por indicação — cria engajamento e reduz churn 18-25%."),
            ("Check-in de satisfação semestral", "Contato proativo a cada 6 meses para garantir satisfação e antecipar problemas."),
            ("Upgrade de velocidade gratuito por 3 meses", "Aumenta percepção de valor sem comprometer margem a longo prazo."),
        ],
        "kpi_meta": "Meta: manter < 15% de migração para tier Alto",
    },
    "Baixo": {
        "titulo": "✅ Manutenção de Relacionamento — Baixo Risco",
        "cor": C['green'],
        "acoes": [
            ("Programa Ambassador", "Convide para grupo VIP — clientes satisfeitos viram defensores da marca."),
            ("Benefício de aniversário", "Presente ou desconto no mês de aniversário do contrato."),
            ("Monitoramento passivo mensal", "Score de risco revisado mensalmente. Qualquer subida → acionar protocolo Médio."),
            ("Upsell consultivo", "Boa hora para oferecer serviços adicionais — cliente está satisfeito e receptivo."),
        ],
        "kpi_meta": "Meta: manter NPS > 70 e score < 0.3",
    },
}

def build_radar_app(data):
    app = dash.Dash(__name__, external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap",
    ], title="Telecom X — Churn Radar", suppress_callback_exceptions=True)

    pred_df = data['pred_df']
    ts      = data['tier_summary']

    def kpi(label, value, color, sub=None):
        return html.Div([
            html.P(label, style={"color": C['sub'], "fontSize": "10px", "letterSpacing": "2px",
                                  "textTransform": "uppercase", "margin": "0 0 6px",
                                  "fontFamily": "'JetBrains Mono'"}),
            html.P(value, style={"color": color, "fontSize": "26px", "fontWeight": "700",
                                  "margin": "0", "fontFamily": "'Syne'"}),
            html.P(sub or "", style={"color": C['sub'], "fontSize": "11px",
                                      "margin": "4px 0 0", "fontFamily": "'JetBrains Mono'"})
        ], style={"background": C['card'], "border": f"1px solid {C['border']}",
                  "borderLeft": f"3px solid {color}", "borderRadius": "8px",
                  "padding": "16px 20px", "flex": "1", "minWidth": "140px"})

    def tier_card(tier):
        color = TIER_COLORS[tier]
        n     = int(ts.loc[tier,'n']) if tier in ts.index else 0
        pct   = n / data['n_active'] * 100
        prob  = float(ts.loc[tier,'prob_media'])*100 if tier in ts.index else 0
        rec   = float(ts.loc[tier,'receita_risco']) if tier in ts.index else 0
        ticket= float(ts.loc[tier,'ticket_medio']) if tier in ts.index else 0
        ten   = float(ts.loc[tier,'tenure_medio']) if tier in ts.index else 0
        return html.Div([
            html.Div([
                html.Span(TIER_EMOJI[tier], style={"fontSize": "22px"}),
                html.Span(f"  TIER {tier.upper()}", style={"fontFamily": "'Syne'", "fontSize": "14px",
                                                            "fontWeight": "700", "color": color,
                                                            "letterSpacing": "2px"}),
            ], style={"marginBottom": "12px", "display": "flex", "alignItems": "center"}),
            html.Div([
                html.P([html.Span("Clientes ", style={"color": C['sub']}),
                        html.Span(f"{n:,}", style={"color": C['text'], "fontWeight": "700", "fontSize": "20px"})],
                        style={"margin": "0 0 4px", "fontFamily": "'JetBrains Mono'"}),
                html.P(f"{pct:.1f}% da base ativa", style={"color": C['sub'], "fontSize": "11px",
                                                            "margin": "0 0 10px", "fontFamily": "'JetBrains Mono'"}),
                _stat_line("Score médio", f"{prob:.0f}%", color),
                _stat_line("Ticket médio", f"R${ticket:.0f}/mês", C['text']),
                _stat_line("Tenure médio", f"{ten:.0f} meses", C['text']),
                _stat_line("Receita em risco", f"R${rec:,.0f}/mês", color),
            ]),
        ], style={"background": C['card2'], "border": f"1px solid {color}33",
                  "borderTop": f"3px solid {color}", "borderRadius": "8px",
                  "padding": "18px", "flex": "1", "minWidth": "180px",
                  "transition": "all 0.2s"})

    def _stat_line(label, value, color):
        return html.Div([
            html.Span(f"{label}: ", style={"color": C['sub'], "fontSize": "11px",
                                            "fontFamily": "'JetBrains Mono'"}),
            html.Span(value, style={"color": color, "fontSize": "12px",
                                     "fontWeight": "600", "fontFamily": "'JetBrains Mono'"}),
        ], style={"marginBottom": "4px"})

    def playbook_card(tier):
        pb = RETENCAO_PLAYBOOK[tier]
        return html.Div([
            html.P(pb['titulo'], style={"color": pb['cor'], "fontFamily": "'Syne'",
                                        "fontSize": "14px", "fontWeight": "700",
                                        "marginBottom": "14px"}),
            *[html.Div([
                html.P(a[0], style={"color": C['text'], "fontWeight": "600",
                                     "fontSize": "12px", "margin": "0 0 3px",
                                     "fontFamily": "'JetBrains Mono'"}),
                html.P(a[1], style={"color": C['sub'], "fontSize": "11px",
                                     "margin": "0 0 10px", "lineHeight": "1.5"}),
              ]) for a in pb['acoes']],
            html.Div(pb['kpi_meta'], style={"color": pb['cor'], "fontSize": "11px",
                                             "fontFamily": "'JetBrains Mono'",
                                             "borderTop": f"1px solid {pb['cor']}44",
                                             "paddingTop": "10px", "marginTop": "4px",
                                             "fontWeight": "600"}),
        ], style={"background": C['card2'], "border": f"1px solid {pb['cor']}44",
                  "borderLeft": f"3px solid {pb['cor']}", "borderRadius": "8px",
                  "padding": "18px", "flex": "1", "minWidth": "230px"})

    def G(fig, h=350, **kw):
        return dcc.Graph(figure=fig, style={"height": f"{h}px"},
                         config={"displayModeBar": False}, **kw)

    def ROW(children, pad="0 36px 18px"):
        return html.Div(children, style={"display": "flex", "gap": "14px", "padding": pad})

    def CARD(children, **style):
        s = {"background": C['card'], "border": f"1px solid {C['border']}",
             "borderRadius": "8px", "padding": "8px"}
        s.update(style)
        return html.Div(children, style=s)

    def section_title(txt, color=C['sub']):
        return html.P(txt, style={"color": color, "fontSize": "10px", "letterSpacing": "2px",
                                   "textTransform": "uppercase", "fontFamily": "'JetBrains Mono'",
                                   "margin": "0 0 14px", "padding": "4px 0"})

    # ── TABELA de clientes ────────────────────────────────────────
    tbl_cols = ['cliente_id','risco_pct','tier','tenure','monthly']
    if 'contrato' in pred_df.columns: tbl_cols.append('contrato')
    tbl_cols.append('acao')
    tbl_data = pred_df[tbl_cols].head(200).to_dict('records')

    tbl_style_data_conditional = [
        {'if': {'filter_query': '{tier} = "Crítico"'},
         'backgroundColor': '#2A0A0A', 'color': C['red']},
        {'if': {'filter_query': '{tier} = "Alto"'},
         'backgroundColor': '#1F1200', 'color': C['orange']},
        {'if': {'filter_query': '{tier} = "Médio"'},
         'backgroundColor': '#1A1600', 'color': C['yellow']},
        {'if': {'filter_query': '{tier} = "Baixo"'},
         'backgroundColor': '#0A1A10', 'color': C['green']},
    ]

    n_critico = int(ts.loc['Crítico','n']) if 'Crítico' in ts.index else 0
    n_alto    = int(ts.loc['Alto','n'])    if 'Alto'    in ts.index else 0
    n_medio   = int(ts.loc['Médio','n'])   if 'Médio'   in ts.index else 0
    n_baixo   = int(ts.loc['Baixo','n'])   if 'Baixo'   in ts.index else 0
    total_risco = float(data['total_risco'])
    pct_risco = (n_critico + n_alto) / data['n_active'] * 100

    app.layout = html.Div(id='full-dashboard',
                          style={"background": C['bg'], "minHeight": "100vh",
                                 "color": C['text'], "fontFamily": "'JetBrains Mono', monospace"}, children=[

        # ── SCREENSHOT (dispara via clientside_callback) ──────────
        dcc.Store(id='screenshot-trigger', data=0),

        # ── HEADER ────────────────────────────────────────────────
        html.Div(style={"background": C['card'], "borderBottom": f"1px solid {C['border']}",
                        "padding": "18px 36px", "display": "flex",
                        "justifyContent": "space-between", "alignItems": "center"}, children=[
            html.Div([
                html.Span("TELECOM X", style={"fontFamily": "'Syne'", "fontWeight": "800",
                                               "fontSize": "22px", "color": C['blue'],
                                               "letterSpacing": "3px"}),
                html.Span("  ⟩  CHURN RADAR", style={"fontFamily": "'Syne'", "color": C['sub'],
                                                      "fontSize": "14px", "letterSpacing": "2px"}),
                html.Div("Identificação de Clientes em Risco + Plano de Retenção",
                         style={"color": C['sub'], "fontSize": "11px", "marginTop": "4px",
                                "fontFamily": "'JetBrains Mono'"}),
                html.Div("Luiz Fernando Barbosa",
                         style={"color": C['sub'], "fontSize": "11px", "marginTop": "4px",
                                "fontFamily": "'JetBrains Mono'"}),
            ]),
            html.Div([
                html.Div([
                    html.Span("⬤ ", style={"color": C['green'], "fontSize": "9px"}),
                    html.Span(f"Modelo Ensemble  AUC={data['auc_ens']:.3f}",
                              style={"color": C['sub'], "fontSize": "11px"}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span("📁 ", style={"fontSize": "11px"}),
                    html.Span(os.path.basename(data['csv_path']),
                              style={"color": C['sub'], "fontSize": "11px"}),
                ], style={"marginBottom": "8px"}),
                html.Button(
                    id='btn-screenshot',
                    children='📸 Capturar Dashboard',
                    n_clicks=0,
                    style={
                        "background": C['card2'],
                        "border": f"1px solid {C['blue']}",
                        "color": C['blue'],
                        "padding": "7px 16px",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                        "fontSize": "12px",
                        "fontFamily": "'JetBrains Mono'",
                        "fontWeight": "600",
                        "letterSpacing": "0.5px",
                    }
                ),
            ], style={"textAlign": "right"}),
        ]),

        # ── KPIs ──────────────────────────────────────────────────
        html.Div(style={"display": "flex", "gap": "12px", "padding": "20px 36px"}, children=[
            kpi("Base Ativa Total",    f"{data['n_active']:,}", C['blue'],  "clientes monitorados"),
            kpi("Risco Crítico 🔴",    f"{n_critico:,}",        C['red'],   f"{n_critico/data['n_active']*100:.1f}% da base"),
            kpi("Risco Alto 🟠",       f"{n_alto:,}",           C['orange'],f"{n_alto/data['n_active']*100:.1f}% da base"),
            kpi("Em Risco (≥50%)",     f"{n_critico+n_alto:,}", C['yellow'],f"{pct_risco:.1f}% precisam atenção"),
            kpi("Receita em Risco",    f"R${total_risco:,.0f}", C['orange'], "/mês ponderado"),
            kpi("Precision do Modelo", f"{data['auc_ens']:.3f}", C['blue'],  "ROC-AUC Ensemble"),
        ]),

        # ── TIER CARDS ─────────────────────────────────────────────
        html.Div(style={"padding": "0 36px 18px"}, children=[
            section_title("Segmentação por Nível de Risco"),
            html.Div([tier_card(t) for t in ['Crítico','Alto','Médio','Baixo']],
                     style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
        ]),

        # ── GRÁFICOS LINHA 1 ──────────────────────────────────────
        ROW([
            CARD(G(fig_radar_donut(data), h=340), **{"flex":"1"}),
            CARD(G(fig_prob_histogram(pred_df), h=340), **{"flex":"2"}),
            CARD(G(fig_receita_risco_tier(data), h=340), **{"flex":"1"}),
        ]),

        # ── GRÁFICOS LINHA 2 ──────────────────────────────────────
        ROW([
            CARD(G(fig_scatter_risk(pred_df), h=400), **{"flex":"3"}),
            CARD(G(fig_tenure_risco(pred_df), h=400), **{"flex":"2"}),
        ]),

        # ── LINHA 3: Importância + filtro ─────────────────────────
        ROW([
            CARD(G(fig_importance(data['rf_imp']), h=380), **{"flex":"2"}),
            html.Div(style={"flex":"1","background":C['card'],"border":f"1px solid {C['border']}",
                            "borderRadius":"8px","padding":"16px 20px"}, children=[
                section_title("Filtrar lista por Tier"),
                dcc.Dropdown(
                    id='tier-filter',
                    options=[{'label': f"{TIER_EMOJI[t]} {t}", 'value': t}
                             for t in ['Crítico','Alto','Médio','Baixo']],
                    value=['Crítico','Alto'],
                    multi=True,
                    style={"background": C['card2'], "color": C['bg']},
                    className='mb-3',
                ),
                html.Div(id='filter-stats', style={"color": C['sub'], "fontSize": "11px",
                                                    "fontFamily": "'JetBrains Mono'",
                                                    "marginTop": "10px"}),
                html.Hr(style={"borderColor": C['border'], "margin": "16px 0"}),
                section_title("Resumo financeiro do filtro"),
                html.Div(id='filter-financeiro'),
            ]),
        ]),

        # ── TABELA DE CLIENTES ────────────────────────────────────
        html.Div(style={"padding": "0 36px 18px"}, children=[
            section_title("Lista de Clientes em Risco — Top 200 (maior score primeiro)"),
            html.Div(id='tabela-clientes', children=[
                dash_table.DataTable(
                    id='risk-table',
                    columns=[{"name": c.replace('_',' ').title(), "id": c} for c in tbl_cols],
                    data=tbl_data,
                    filter_action="native",
                    sort_action="native",
                    page_size=15,
                    style_table={"overflowX": "auto"},
                    style_cell={"backgroundColor": C['card2'], "color": C['text'],
                                 "border": f"1px solid {C['border']}", "padding": "8px 12px",
                                 "fontFamily": "'JetBrains Mono'", "fontSize": "12px"},
                    style_header={"backgroundColor": C['card'], "color": C['sub'],
                                   "fontWeight": "600", "border": f"1px solid {C['border']}",
                                   "letterSpacing": "1px"},
                    style_data_conditional=tbl_style_data_conditional,
                )
            ]),
        ]),

        # ── PLAYBOOK DE RETENÇÃO ──────────────────────────────────
        html.Div(style={"padding": "0 36px 18px"}, children=[
            section_title("Playbook de Retenção — Ações por Tier", C['cyan']),
            html.Div([playbook_card(t) for t in ['Crítico','Alto','Médio','Baixo']],
                     style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
        ]),

        # ── FOOTER ────────────────────────────────────────────────
        html.Div(style={"background": C['card'], "borderTop": f"1px solid {C['border']}",
                        "padding": "16px 36px", "display": "flex",
                        "justifyContent": "space-between", "alignItems": "center"}, children=[
            html.Span(f"Telecom X — Churn Radar  ·  {data['n_active']:,} clientes analisados  "
                      f"·  Ensemble RF+GBM+LR  AUC={data['auc_ens']:.3f}",
                      style={"color": C['sub'], "fontSize": "11px", "fontFamily": "'JetBrains Mono'"}),
            html.Span(f"Lista salva em: {os.path.basename(data['out_path'])}",
                      style={"color": C['green'], "fontSize": "11px", "fontFamily": "'JetBrains Mono'"}),
        ]),
    ])

    # ── CALLBACKS ─────────────────────────────────────────────────
    @app.callback(
        [Output('filter-stats','children'), Output('filter-financeiro','children')],
        Input('tier-filter','value')
    )
    def update_filter_info(tiers):
        if not tiers:
            return "Nenhum tier selecionado", ""
        sub = pred_df[pred_df['tier'].isin(tiers)]
        n   = len(sub)
        rec = sub['receita_risco'].sum()
        ticket_m = sub['monthly'].mean()
        ten_m    = sub['tenure'].mean()

        stats = html.Div([
            html.P(f"Clientes selecionados: {n:,}", style={"color": C['text'], "margin": "4px 0"}),
            html.P(f"% da base ativa: {n/data['n_active']*100:.1f}%", style={"color": C['sub'], "margin": "4px 0"}),
            html.P(f"Score médio: {sub['prob_churn'].mean()*100:.1f}%", style={"color": C['yellow'], "margin": "4px 0"}),
            html.P(f"Tenure médio: {ten_m:.0f} meses", style={"color": C['sub'], "margin": "4px 0"}),
        ])
        fin = html.Div([
            html.P(f"Receita em risco: R${rec:,.0f}/mês", style={"color": C['orange'], "fontWeight": "700",
                                                                    "fontFamily": "'Syne'", "fontSize": "15px"}),
            html.P(f"Ticket médio: R${ticket_m:.0f}/mês", style={"color": C['sub'], "margin": "4px 0"}),
            html.P(f"Se reter 50%: +R${rec*0.5:,.0f}/mês", style={"color": C['green'], "fontWeight": "600",
                                                                     "margin": "8px 0 4px"}),
            html.P(f"ROI anual: R${rec*0.5*12:,.0f}/ano", style={"color": C['green']}),
        ])
        return stats, fin

    # ── SCREENSHOT — clientside callback (sem onClick no Button) ──
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks || n_clicks === 0) return window.dash_clientside.no_update;

            var btn = document.getElementById('btn-screenshot');
            if (btn) { btn.innerText = '⏳ Capturando...'; btn.disabled = true; }

            function doCapture() {
                html2canvas(document.getElementById('full-dashboard'), {
                    backgroundColor: '#080C10',
                    scale: 1.5,
                    useCORS: true,
                    allowTaint: true,
                    logging: false,
                    windowWidth: document.documentElement.scrollWidth,
                    windowHeight: document.documentElement.scrollHeight,
                }).then(function(canvas) {
                    var link = document.createElement('a');
                    var ts = new Date().toISOString().slice(0,19).replace(/[T:]/g, '-');
                    link.download = 'telecom_x_churn_radar_' + ts + '.png';
                    link.href = canvas.toDataURL('image/png');
                    link.click();
                    if (btn) { btn.innerText = '📸 Capturar Dashboard'; btn.disabled = false; }
                }).catch(function(e) {
                    alert('Erro na captura: ' + e.message);
                    if (btn) { btn.innerText = '📸 Capturar Dashboard'; btn.disabled = false; }
                });
            }

            if (typeof html2canvas === 'undefined') {
                var script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
                script.onload = doCapture;
                document.head.appendChild(script);
            } else {
                doCapture();
            }

            return window.dash_clientside.no_update;
        }
        """,
        Output('screenshot-trigger', 'data'),
        Input('btn-screenshot', 'n_clicks'),
        prevent_initial_call=True,
    )

    return app


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    csv_path = next((a for a in sys.argv[1:] if not a.startswith('--')), INPUT_CSV)

    if not os.path.exists(csv_path):
        print(f"\n❌  Arquivo não encontrado: {csv_path}")
        print(f"    Edite INPUT_CSV ou: python pyTelecomXChurnRadar.py seu_arquivo.csv")
        sys.exit(1)

    data = run_radar_pipeline(csv_path)
    app  = build_radar_app(data)
    app.run(debug=False, host="0.0.0.0", port=8050)