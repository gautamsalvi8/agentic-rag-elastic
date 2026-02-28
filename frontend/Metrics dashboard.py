"""
pages/📊_Metrics_Dashboard.py
---------------------------------
Performance metrics dashboard for RAG system
Shows real-time and aggregate performance statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')))

from metrics_logger import MetricsLogger

st.set_page_config(
    page_title="Metrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize logger
@st.cache_resource
def get_logger():
    return MetricsLogger("metrics_log.json")

logger = get_logger()

# ==================== HEADER ====================
st.title("📊 System Performance Dashboard")
st.markdown("Real-time metrics and performance analytics for ElasticNode AI")
st.markdown("---")

# ==================== SUMMARY CARDS ====================
stats = logger.get_stats()

if stats['total_queries'] == 0:
    st.info("🔍 No queries logged yet. Start using the chat to see metrics here!")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Queries",
        f"{stats['total_queries']}",
        help="Total number of queries processed"
    )

with col2:
    st.metric(
        "Avg Latency",
        f"{stats['avg_latency']:.2f}s",
        delta=f"{stats['median_latency']:.2f}s median",
        help="Average end-to-end query latency"
    )

with col3:
    cache_perf = logger.get_cache_performance()
    st.metric(
        "Cache Hit Rate",
        f"{cache_perf['cache_hit_rate']:.1f}%",
        delta=f"{cache_perf['total_hits']} hits",
        help="Percentage of queries served from cache"
    )

with col4:
    st.metric(
        "Avg Top Score",
        f"{stats['avg_top_score']:.2f}",
        help="Average relevance score of top result"
    )

with col5:
    st.metric(
        "P95 Latency",
        f"{stats['p95_latency']:.2f}s",
        help="95th percentile latency (worst 5% of queries)"
    )

st.markdown("---")

# ==================== LATENCY BREAKDOWN ====================
st.subheader("⚡ Latency Breakdown")

col1, col2 = st.columns(2)

with col1:
    # Pie chart of time distribution
    breakdown_data = {
        "Component": ["Search", "Rerank", "Embedding", "Other"],
        "Time": [
            stats['avg_search_time'],
            stats['avg_rerank_time'],
            stats['avg_embedding_time'],
            max(0, stats['avg_latency'] - stats['avg_search_time'] - stats['avg_rerank_time'] - stats['avg_embedding_time'])
        ]
    }
    
    fig_pie = px.pie(
        breakdown_data,
        values="Time",
        names="Component",
        title="Average Time Distribution",
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Bar chart of absolute times
    fig_bar = go.Figure(data=[
        go.Bar(name='Search', x=['Time (s)'], y=[stats['avg_search_time']]),
        go.Bar(name='Rerank', x=['Time (s)'], y=[stats['avg_rerank_time']]),
        go.Bar(name='Embedding', x=['Time (s)'], y=[stats['avg_embedding_time']])
    ])
    fig_bar.update_layout(
        title="Component Latencies",
        barmode='group',
        yaxis_title="Seconds"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ==================== LATENCY OVER TIME ====================
st.subheader("📈 Performance Over Time")

# Get recent metrics for timeline
recent = logger.get_recent_metrics(50)  # Last 50 queries

if recent:
    df = pd.DataFrame(recent)
    df['query_number'] = range(1, len(df) + 1)
    
    # Line chart of latency over time
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=df['query_number'],
        y=df['total_latency'],
        mode='lines+markers',
        name='Total Latency',
        line=dict(color='#667eea', width=2),
        marker=dict(
            size=8,
            color=df['cached'].apply(lambda x: '#48bb78' if x else '#667eea'),
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>Query %{x}</b><br>Latency: %{y:.3f}s<br>Cached: %{marker.color}<extra></extra>'
    ))
    
    # Add average line
    fig_timeline.add_hline(
        y=stats['avg_latency'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {stats['avg_latency']:.2f}s"
    )
    
    fig_timeline.update_layout(
        title="Latency Timeline (Last 50 Queries)",
        xaxis_title="Query Number",
        yaxis_title="Latency (seconds)",
        hovermode='closest',
        showlegend=True
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Show color legend
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟢 **Green** = Cached query (fast)")
    with col2:
        st.markdown("🔵 **Blue** = Fresh query (slower)")

st.markdown("---")

# ==================== CACHE PERFORMANCE ====================
st.subheader("💾 Cache Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Cache Hits",
        f"{cache_perf['total_hits']}",
        help="Number of queries served from cache"
    )

with col2:
    st.metric(
        "Cache Misses",
        f"{cache_perf['total_misses']}",
        help="Number of queries that required fresh search"
    )

with col3:
    if cache_perf['avg_fresh_latency'] > 0:
        speedup = (1 - cache_perf['avg_cached_latency']/cache_perf['avg_fresh_latency']) * 100
        st.metric(
            "Speed Improvement",
            f"{speedup:.1f}%",
            help="How much faster cached queries are"
        )
    else:
        st.metric("Speed Improvement", "N/A")

# Cache comparison chart
if cache_perf['total_hits'] > 0 and cache_perf['total_misses'] > 0:
    cache_comparison = pd.DataFrame({
        "Type": ["Cached", "Fresh"],
        "Avg Latency": [cache_perf['avg_cached_latency'], cache_perf['avg_fresh_latency']],
        "Count": [cache_perf['total_hits'], cache_perf['total_misses']]
    })
    
    fig_cache = px.bar(
        cache_comparison,
        x="Type",
        y="Avg Latency",
        color="Type",
        text="Count",
        title="Cached vs Fresh Query Performance",
        color_discrete_map={"Cached": "#48bb78", "Fresh": "#667eea"}
    )
    fig_cache.update_traces(texttemplate='%{text} queries', textposition='outside')
    fig_cache.update_layout(showlegend=False, yaxis_title="Average Latency (s)")
    st.plotly_chart(fig_cache, use_container_width=True)

st.markdown("---")

# ==================== SLOW QUERIES ====================
st.subheader("🐌 Slow Queries (>3s)")

slow_queries = logger.get_slow_queries(threshold=3.0)

if slow_queries:
    st.warning(f"⚠️ Found {len(slow_queries)} slow queries")
    
    slow_df = pd.DataFrame(slow_queries)
    slow_df = slow_df.sort_values('total_latency', ascending=False)
    
    # Show top 10 slowest
    st.dataframe(
        slow_df[['query', 'total_latency', 'search_time', 'rerank_time', 'embedding_time']].head(10),
        use_container_width=True,
        column_config={
            "query": "Query",
            "total_latency": st.column_config.NumberColumn("Total (s)", format="%.3f"),
            "search_time": st.column_config.NumberColumn("Search (s)", format="%.3f"),
            "rerank_time": st.column_config.NumberColumn("Rerank (s)", format="%.3f"),
            "embedding_time": st.column_config.NumberColumn("Embed (s)", format="%.3f"),
        }
    )
else:
    st.success("✅ No slow queries found! All queries under 3 seconds.")

st.markdown("---")

# ==================== RECENT QUERIES ====================
st.subheader("🔍 Recent Queries")

recent_queries = logger.get_recent_metrics(20)

if recent_queries:
    recent_df = pd.DataFrame(recent_queries)
    recent_df = recent_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(
        recent_df[['timestamp', 'query', 'total_latency', 'top_score', 'cached']],
        use_container_width=True,
        column_config={
            "timestamp": "Time",
            "query": "Query",
            "total_latency": st.column_config.NumberColumn("Latency (s)", format="%.3f"),
            "top_score": st.column_config.NumberColumn("Top Score", format="%.2f"),
            "cached": st.column_config.CheckboxColumn("Cached")
        }
    )

st.markdown("---")

# ==================== EXPORT & ACTIONS ====================
st.subheader("📥 Export & Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Export Full Report", use_container_width=True):
        filepath = logger.export_report()
        st.success(f"✅ Exported to {filepath}")

with col2:
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

with col3:
    if st.button("🗑️ Clear Metrics", use_container_width=True, type="secondary"):
        if st.session_state.get('confirm_clear'):
            logger.clear_metrics()
            st.success("✅ Metrics cleared!")
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Click again to confirm")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Performance metrics auto-update with each query</p>
        <p style='font-size: 0.8rem;'>Last updated: {}</p>
    </div>
    """.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
    unsafe_allow_html=True
)