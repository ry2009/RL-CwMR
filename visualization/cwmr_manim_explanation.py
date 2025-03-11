from manim import *
import numpy as np
import pandas as pd

class CWMRRLExplanation(Scene):
    def construct(self):
        # Title and introduction
        title = Text("Confidence-Weighted Mean Reversion with RL", font_size=40)
        subtitle = Text("Advanced Portfolio Optimization Strategies", font_size=30).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Project overview
        project_outline = BulletedList(
            "Problem: Portfolio allocation in financial markets",
            "Approach: Enhance CWMR algorithm with RL techniques",
            "Multiple strategies: Traditional, GRPO, PPO, Multi-agent",
            "Results: Significantly improved risk-adjusted returns",
            font_size=28
        )
        project_outline.next_to(subtitle, DOWN, buff=0.7)
        
        self.play(Write(project_outline), run_time=3)
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(project_outline))
        
        # Part 1: Problem Statement and Traditional Approaches
        part1_title = Text("Part 1: Portfolio Optimization Challenges", font_size=36)
        self.play(Write(part1_title))
        self.wait(1)
        self.play(part1_title.animate.to_edge(UP))
        
        # Portfolio optimization challenges
        challenges = BulletedList(
            "Market uncertainty and volatility",
            "Non-stationary return distributions",
            "Transaction costs erode performance",
            "Balancing risk and return tradeoffs",
            font_size=26
        )
        challenges.next_to(part1_title, DOWN, buff=0.5)
        
        self.play(Write(challenges), run_time=3)
        self.wait(2)
        
        # Traditional approaches
        trad_title = Text("Traditional Approaches", font_size=30)
        trad_title.next_to(challenges, DOWN, buff=0.7)
        
        trad_approaches = BulletedList(
            "Equal Weight: Simple but suboptimal",
            "Mean-Variance Optimization: Sensitive to estimation errors",
            "CWMR: Better for online portfolio selection, but static parameters",
            font_size=24
        )
        trad_approaches.next_to(trad_title, DOWN, buff=0.3)
        
        self.play(Write(trad_title))
        self.play(Write(trad_approaches), run_time=3)
        self.wait(2)
        
        self.play(FadeOut(challenges), FadeOut(trad_title), FadeOut(trad_approaches))
        
        # Mean Reversion Explanation
        mr_def = Text("Mean Reversion: Prices tend to revert to their historical mean", font_size=28)
        mr_def.next_to(part1_title, DOWN, buff=0.5)
        
        # Create a price chart demonstrating mean reversion
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 2, 0.5],
            axis_config={"include_tip": False},
        ).scale(0.7)
        
        axes_labels = axes.get_axis_labels(x_label="Time", y_label="Price")
        
        # Generate mean-reverting price data
        np.random.seed(42)  # For reproducibility
        price = 1.0
        prices = [price]
        mean = 1.0
        for _ in range(99):
            # Mean-reverting random walk
            price += 0.03 * (mean - price) + 0.02 * np.random.randn()
            prices.append(price)
        
        # Create graph of price movement
        price_graph = axes.plot_line_graph(
            x_values=list(range(100)),
            y_values=prices,
            line_color=BLUE,
            add_vertex_dots=False
        )
        
        # Add mean line
        mean_line = DashedLine(
            axes.c2p(0, mean),
            axes.c2p(100, mean),
            color=RED
        )
        
        mean_label = Text("Mean", font_size=20, color=RED).next_to(mean_line, LEFT)
        
        # Highlight mean reversion with arrows
        reversion_arrows = []
        arrow_positions = [20, 40, 60, 80]
        for pos in arrow_positions:
            if prices[pos] > mean:
                arrow = Arrow(
                    axes.c2p(pos, prices[pos]), 
                    axes.c2p(pos, mean), 
                    color=GREEN, 
                    tip_length=0.2
                )
            else:
                arrow = Arrow(
                    axes.c2p(pos, prices[pos]), 
                    axes.c2p(pos, mean), 
                    color=GREEN, 
                    tip_length=0.2
                )
            reversion_arrows.append(arrow)
        
        chart_group = VGroup(axes, axes_labels)
        
        self.play(Write(mr_def))
        self.play(Create(chart_group))
        self.play(Create(price_graph), run_time=2)
        self.play(Create(mean_line), Write(mean_label))
        
        for arrow in reversion_arrows:
            self.play(Create(arrow), run_time=0.5)
        
        self.wait(2)
        
        # CWMR Algorithm Explanation
        cwmr_title = Text("CWMR: Confidence-Weighted Mean Reversion", font_size=28)
        cwmr_title.next_to(chart_group, DOWN, buff=0.7)
        
        cwmr_eq = MathTex(
            r"\mathbf{q}_{t+1} = \arg\min_{\mathbf{q}} D_{KL}(\mathbf{q}||\mathbf{q}_t) \,\, \text{s.t.} \,\, \mathbb{P}_{\mathbf{q}}[1 - \mathbf{q} \cdot \mathbf{x}_{t+1} \geq \varepsilon] \leq \delta"
        ).scale(0.8)
        cwmr_eq.next_to(cwmr_title, DOWN)
        
        # Create bullet points as individual Text objects
        cwmr_points = VGroup(
            Text("• Maintains a Gaussian distribution over portfolio weights", font_size=24),
            Text("• Updates weights based on confidence-weighted price predictions", font_size=24),
            Text("• Controlled by confidence bound (δ) and epsilon (ε) parameters", font_size=24),
            Text("• Projects distribution onto the probability simplex", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT)
        cwmr_points.next_to(cwmr_eq, DOWN)
        
        self.play(Write(cwmr_title))
        self.play(Write(cwmr_eq))
        self.play(Write(cwmr_points), run_time=3)
        self.wait(3)
        
        self.play(
            FadeOut(part1_title), 
            FadeOut(mr_def),
            FadeOut(chart_group), 
            FadeOut(price_graph), 
            FadeOut(mean_line), 
            FadeOut(mean_label), 
            *[FadeOut(arrow) for arrow in reversion_arrows],
            FadeOut(cwmr_title),
            FadeOut(cwmr_eq),
            FadeOut(cwmr_points)
        )
        
        # Part 2: Our RL Enhancement Approach
        part2_title = Text("Part 2: Reinforcement Learning Enhancement", font_size=36)
        self.play(Write(part2_title))
        self.wait(1)
        self.play(part2_title.animate.to_edge(UP))
        
        # RL Framework
        rl_subtitle = Text("RL Framework for Portfolio Optimization", font_size=28)
        rl_subtitle.next_to(part2_title, DOWN, buff=0.5)
        self.play(Write(rl_subtitle))
        
        # Create a detailed RL diagram using Manim
        agent = Circle(radius=0.7, color=BLUE, fill_opacity=0.5)
        agent_label = Text("RL Agent", font_size=20).next_to(agent, DOWN, buff=0.2)
        agent_group = VGroup(agent, agent_label).move_to(3*LEFT)
        
        environment = Rectangle(height=4, width=6, color=GREEN, fill_opacity=0.2)
        env_label = Text("Financial Market Environment", font_size=20).next_to(environment, UP, buff=0.2)
        env_group = VGroup(environment, env_label).move_to(3*RIGHT)
        
        # Inside the agent (policy and value networks)
        policy_net = Rectangle(height=0.4, width=0.8, color=YELLOW, fill_opacity=0.8).move_to(agent.get_center() + UP*0.2)
        policy_label = Text("Policy Net", font_size=12).next_to(policy_net, UP, buff=0.1)
        
        value_net = Rectangle(height=0.4, width=0.8, color=ORANGE, fill_opacity=0.8).move_to(agent.get_center() + DOWN*0.2)
        value_label = Text("Value Net", font_size=12).next_to(value_net, DOWN, buff=0.1)
        
        # Inside the environment (portfolio, market data)
        portfolio_box = Rectangle(height=1, width=1.5, color=BLUE_C, fill_opacity=0.3).move_to(environment.get_center() + UP*0.7)
        portfolio_label = Text("Portfolio", font_size=16).next_to(portfolio_box, UP, buff=0.1)
        
        market_box = Rectangle(height=1, width=1.5, color=RED_C, fill_opacity=0.3).move_to(environment.get_center() + DOWN*0.7)
        market_label = Text("Market Data", font_size=16).next_to(market_box, DOWN, buff=0.1)
        
        # Arrows for the RL loop
        action_arrow = Arrow(agent.get_right(), environment.get_left(), color=YELLOW, buff=0.3)
        action_text = Text("Actions:\nPortfolio Weights", font_size=16).next_to(action_arrow, UP, buff=0.1)
        
        reward_arrow = Arrow(environment.get_top() + LEFT*2, agent.get_top() + RIGHT*2, color=RED, buff=0.3)
        reward_text = Text("Reward:\nPortfolio Return", font_size=16).next_to(reward_arrow, UP, buff=0.1)
        
        state_arrow = Arrow(environment.get_bottom() + LEFT*2, agent.get_bottom() + RIGHT*2, color=PURPLE, buff=0.3)
        state_text = Text("State:\nMarket Features", font_size=16).next_to(state_arrow, DOWN, buff=0.1)
        
        # Create the full diagram
        rl_diagram = VGroup(
            agent_group, env_group,
            policy_net, policy_label, value_net, value_label,
            portfolio_box, portfolio_label, market_box, market_label,
            action_arrow, action_text, reward_arrow, reward_text, state_arrow, state_text
        )
        
        self.play(Create(agent_group))
        self.play(Create(env_group))
        self.play(Create(policy_net), Write(policy_label), Create(value_net), Write(value_label))
        self.play(Create(portfolio_box), Write(portfolio_label), Create(market_box), Write(market_label))
        self.play(Create(action_arrow), Write(action_text))
        self.play(Create(reward_arrow), Write(reward_text))
        self.play(Create(state_arrow), Write(state_text))
        self.wait(2)
        
        # Our RL enhancements
        rl_diagram.scale(0.8).to_edge(UP, buff=1.5)
        
        enhancements_title = Text("Our RL Enhancements to CWMR", font_size=28)
        enhancements_title.next_to(rl_diagram, DOWN, buff=0.5)
        
        enhancements = BulletedList(
            "Dynamic Parameter Adjustment: Adapts to changing market conditions",
            "Transaction Cost Optimization: Balances trading and holding",
            "Feature Engineering: Technical indicators for better state representation",
            "Risk-Aware Rewards: Incorporates Sharpe ratio and drawdown metrics",
            font_size=24
        )
        enhancements.next_to(enhancements_title, DOWN, buff=0.3)
        
        self.play(Write(enhancements_title))
        self.play(Write(enhancements), run_time=3)
        self.wait(2)
        
        # Specific Algorithm Implementations
        algorithms_title = Text("Implemented Strategy Variants", font_size=28)
        algorithms_title.next_to(enhancements, DOWN, buff=0.5)
        
        algorithms = VGroup(
            BulletedList(
                "GRPO-CWMR: Group Relativity Policy Optimization",
                "- Groups similar stocks for hierarchical optimization",
                "- Showed highest Sharpe ratio (0.7)",
                font_size=22
            ),
            BulletedList(
                "RL-PPO: Proximal Policy Optimization",
                "- Standard PPO algorithm with custom rewards",
                "- Balanced performance (0.6 Sharpe)",
                font_size=22
            ),
            BulletedList(
                "Multi-Agent Ensemble:",
                "- Multiple CWMR agents with different parameters",
                "- Best risk-adjusted performance (0.67 Sharpe, 36% max drawdown)",
                font_size=22
            )
        ).arrange(RIGHT, buff=0.3).next_to(algorithms_title, DOWN, buff=0.3)
        
        self.play(Write(algorithms_title))
        self.play(Write(algorithms), run_time=3)
        self.wait(3)
        
        # Transition to Part 3
        self.play(
            FadeOut(part2_title),
            FadeOut(rl_subtitle),
            FadeOut(rl_diagram),
            FadeOut(enhancements_title),
            FadeOut(enhancements),
            FadeOut(algorithms_title),
            FadeOut(algorithms)
        )
        
        # Part 3: Results and Comparison
        part3_title = Text("Part 3: Strategy Performance Analysis", font_size=36)
        self.play(Write(part3_title))
        self.wait(1)
        self.play(part3_title.animate.to_edge(UP))
        
        # Performance metrics table
        results_subtitle = Text("Comparative Performance Metrics", font_size=28)
        results_subtitle.next_to(part3_title, DOWN, buff=0.5)
        self.play(Write(results_subtitle))
        
        table = MobjectTable(
            [
                [Text("Strategy", font_size=20), Text("Sharpe Ratio", font_size=20), 
                 Text("Max Drawdown", font_size=20), Text("Return", font_size=20)],
                [Text("GRPO-CWMR", font_size=20), Text("0.70", font_size=20, color=GREEN), 
                 Text("47%", font_size=20, color=RED), Text("+++", font_size=20, color=GREEN)],
                [Text("RL-PPO", font_size=20), Text("0.60", font_size=20, color=YELLOW), 
                 Text("42%", font_size=20, color=YELLOW), Text("++", font_size=20, color=YELLOW)],
                [Text("Multi-Agent", font_size=20), Text("0.67", font_size=20, color=GREEN), 
                 Text("36%", font_size=20, color=GREEN), Text("+++", font_size=20, color=GREEN)],
                [Text("Equal Weight", font_size=20), Text("N/A", font_size=20), 
                 Text("N/A", font_size=20), Text("6%", font_size=20, color=RED)],
                [Text("Original CWMR", font_size=20), Text("N/A", font_size=20), 
                 Text("N/A", font_size=20), Text("12-13%", font_size=20, color=YELLOW)]
            ],
            include_outer_lines=True
        ).scale(0.8)
        
        table.next_to(results_subtitle, DOWN, buff=0.5)
        self.play(Create(table), run_time=2)
        self.wait(2)
        
        # Highlight key insights
        insights_title = Text("Key Insights", font_size=28)
        insights_title.next_to(table, DOWN, buff=0.7)
        
        # Create insights as individual Text objects
        insights = VGroup(
            Text("• All RL-enhanced strategies outperform traditional approaches", font_size=24),
            Text("• Multi-agent ensemble provides best balance of return and risk", font_size=24),
            Text("• 36% max drawdown (Multi-agent) vs 47% (GRPO) is significant improvement", font_size=24),
            Text("• Higher Sharpe ratios indicate better risk-adjusted performance", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.play(Write(insights_title))
        self.play(Write(insights), run_time=3)
        self.wait(3)
        
        # Performance visualization
        performance_viz_title = Text("Portfolio Value Evolution", font_size=28)
        
        # Create a dummy chart showing portfolio value over time
        chart_axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0.5, 1.5, 0.25],
            axis_config={"include_tip": False},
            x_axis_config={"include_numbers": True},
            y_axis_config={"include_numbers": True},
        ).scale(0.7)
        
        chart_labels = chart_axes.get_axis_labels(x_label="Trading Days", y_label="Portfolio Value")
        
        # Generate performance curves for each strategy
        # Multi-agent (best performance)
        multi_agent_curve = chart_axes.plot(
            lambda x: 1 + 0.0035*x - 0.0000075*x**2,
            x_range=[0, 100],
            color=GREEN,
        )
        
        # GRPO (higher volatility)
        np.random.seed(42)
        grpo_values = [1.0]
        for i in range(1, 101):
            next_val = grpo_values[-1] * (1 + 0.004*i - 0.00001*i**2 + 0.015*np.random.randn())
            grpo_values.append(next_val)
            
        grpo_curve = chart_axes.plot_line_graph(
            x_values=list(range(101)),
            y_values=grpo_values,
            line_color=BLUE,
            add_vertex_dots=False
        )
        
        # PPO (medium performance)
        ppo_curve = chart_axes.plot(
            lambda x: 1 + 0.003*x - 0.000007*x**2,
            x_range=[0, 100],
            color=YELLOW,
        )
        
        # Equal weight (poor performance)
        equal_weight_curve = chart_axes.plot(
            lambda x: 1 + 0.0008*x - 0.000002*x**2,
            x_range=[0, 100],
            color=RED,
        )
        
        # Create labels
        multi_agent_label = Text("Multi-Agent", font_size=16, color=GREEN).next_to(chart_axes, RIGHT, buff=0.3).shift(UP*0.5)
        grpo_label = Text("GRPO", font_size=16, color=BLUE).next_to(chart_axes, RIGHT, buff=0.3).shift(UP*0.2)
        ppo_label = Text("PPO", font_size=16, color=YELLOW).next_to(chart_axes, RIGHT, buff=0.3).shift(DOWN*0.1)
        equal_weight_label = Text("Equal Weight", font_size=16, color=RED).next_to(chart_axes, RIGHT, buff=0.3).shift(DOWN*0.4)
        
        # Create a line for each label
        multi_agent_line = Line(multi_agent_label.get_left(), multi_agent_label.get_left() - LEFT*0.3, color=GREEN)
        grpo_line = Line(grpo_label.get_left(), grpo_label.get_left() - LEFT*0.3, color=BLUE)
        ppo_line = Line(ppo_label.get_left(), ppo_label.get_left() - LEFT*0.3, color=YELLOW)
        equal_weight_line = Line(equal_weight_label.get_left(), equal_weight_label.get_left() - LEFT*0.3, color=RED)
        
        # Group everything for the chart
        performance_chart = VGroup(
            chart_axes, chart_labels,
            multi_agent_curve, grpo_curve, ppo_curve, equal_weight_curve,
            multi_agent_label, grpo_label, ppo_label, equal_weight_label,
            multi_agent_line, grpo_line, ppo_line, equal_weight_line
        )
        
        # Position and animate
        self.play(FadeOut(table), FadeOut(insights_title), FadeOut(insights))
        
        performance_viz_title.next_to(results_subtitle, DOWN, buff=0.5)
        performance_chart.next_to(performance_viz_title, DOWN, buff=0.3)
        
        self.play(Write(performance_viz_title))
        self.play(Create(chart_axes), Create(chart_labels))
        self.play(Create(equal_weight_curve), Create(equal_weight_label), Create(equal_weight_line))
        self.play(Create(ppo_curve), Create(ppo_label), Create(ppo_line))
        self.play(Create(grpo_curve), Create(grpo_label), Create(grpo_line))
        self.play(Create(multi_agent_curve), Create(multi_agent_label), Create(multi_agent_line))
        self.wait(3)
        
        # Highlight multi-agent advantage
        highlight_rect = SurroundingRectangle(multi_agent_label, color=GREEN, buff=0.1)
        highlight_text = Text("Best Risk-Adjusted Performance", font_size=24, color=GREEN).next_to(performance_chart, DOWN, buff=0.5)
        
        self.play(Create(highlight_rect))
        self.play(Write(highlight_text))
        self.wait(2)
        
        # Transition to Part 4
        self.play(
            FadeOut(part3_title),
            FadeOut(results_subtitle),
            FadeOut(performance_viz_title),
            FadeOut(performance_chart),
            FadeOut(highlight_rect),
            FadeOut(highlight_text)
        )
        
        # Part 4: Implications and Future Work
        part4_title = Text("Part 4: Implications and Future Work", font_size=36)
        self.play(Write(part4_title))
        self.wait(1)
        self.play(part4_title.animate.to_edge(UP))
        
        # Key takeaways
        takeaways_title = Text("Key Takeaways", font_size=28)
        takeaways_title.next_to(part4_title, DOWN, buff=0.5)
        
        takeaways = BulletedList(
            "RL successfully enhances traditional trading algorithms",
            "Multi-agent approaches provide robustness across different market conditions",
            "Risk management improves with adaptive strategies",
            "Transaction costs can be optimized automatically",
            font_size=24
        )
        takeaways.next_to(takeaways_title, DOWN, buff=0.3)
        
        self.play(Write(takeaways_title))
        self.play(Write(takeaways), run_time=3)
        self.wait(2)
        
        # Future research directions
        future_title = Text("Future Research Directions", font_size=28)
        future_title.next_to(takeaways, DOWN, buff=0.7)
        
        future_work = BulletedList(
            "Advanced RL Architectures: Transformers, Graph Neural Networks",
            "Multi-agent Cooperation and Competition",
            "Explainable AI for Investment Decisions",
            "Adaptive Risk Management in Different Market Regimes",
            "Integration with Alternative Data and Sentiment Analysis",
            font_size=24
        )
        future_work.next_to(future_title, DOWN, buff=0.3)
        
        self.play(Write(future_title))
        self.play(Write(future_work), run_time=3)
        self.wait(2)
        
        # Limitations and practical considerations
        limitations_title = Text("Limitations and Practical Considerations", font_size=28)
        limitations_title.next_to(future_work, DOWN, buff=0.7)
        
        limitations = BulletedList(
            "Data limitations: Tested on finite historical data",
            "Computational complexity: RL training can be resource-intensive",
            "Overfitting risks: Need for robust validation protocols",
            "Market impact: Strategy effectiveness may decay with scale",
            font_size=24
        )
        limitations.next_to(limitations_title, DOWN, buff=0.3)
        
        self.play(Write(limitations_title))
        self.play(Write(limitations), run_time=3)
        self.wait(3)
        
        # Conclusion
        self.play(
            FadeOut(part4_title),
            FadeOut(takeaways_title),
            FadeOut(takeaways),
            FadeOut(future_title),
            FadeOut(future_work),
            FadeOut(limitations_title),
            FadeOut(limitations)
        )
        
        conclusion_title = Text("Conclusion", font_size=40)
        self.play(Write(conclusion_title))
        self.wait(1)
        self.play(conclusion_title.animate.to_edge(UP))
        
        # Create conclusion points as individual Text objects
        conclusion_points = VGroup(
            Text("• Multi-agent CWMR ensemble achieves superior risk-adjusted returns", font_size=24),
            Text("• 0.67 Sharpe ratio with only 36% maximum drawdown", font_size=24),
            Text("• Outperforms traditional approaches by incorporating adaptive learning", font_size=24),
            Text("• Framework provides foundation for further research and enhancement", font_size=24),
            Text("• Demonstrates the value of combining traditional finance with modern ML", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT)
        conclusion_points.next_to(conclusion_title, DOWN, buff=0.5)
        
        self.play(Write(conclusion_points), run_time=3)
        self.wait(3)
        
        # Final credits and repository information
        self.play(FadeOut(conclusion_title), FadeOut(conclusion_points))
        
        final_title = Text("CWMR-RL: Advanced Portfolio Optimization", font_size=40, color=BLUE)
        github_text = Text("github.com/ry2009/RL-CwMR", font_size=30, color=YELLOW).next_to(final_title, DOWN, buff=0.5)
        contact_text = Text("For more information and collaboration opportunities", font_size=24).next_to(github_text, DOWN, buff=0.5)
        
        self.play(Write(final_title))
        self.play(Write(github_text))
        self.play(Write(contact_text))
        self.wait(3)
        
        # Final fade out
        self.play(FadeOut(final_title), FadeOut(github_text), FadeOut(contact_text)) 