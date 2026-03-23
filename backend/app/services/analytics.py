from datetime import datetime, timedelta

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.models import Game, GameRequest, PlaySession, GameAnalytics, User, now_ist


class AnalyticsService:
    
    @staticmethod
    def record_play_end(db: Session, session: PlaySession):
        session.ended_at = now_ist()
        
        game = db.query(Game).filter(Game.id == session.game_id).first()
        if game:
            game.total_plays = Game.total_plays + 1
            if session.state == "WIN":
                game.total_wins = Game.total_wins + 1
        
        db.commit()
    
    @staticmethod
    def get_game_analytics(db: Session, game_db_id: str, days: int = 30) -> dict:
        game = db.query(Game).filter(Game.id == game_db_id).first()
        if not game:
            return {}
        
        since = datetime.now() - timedelta(days=days)
        
        sessions = (
            db.query(PlaySession)
            .filter(
                and_(
                    PlaySession.game_id == game_db_id,
                    PlaySession.started_at >= since,
                )
            )
            .all()
        )
        
        total = len(sessions)
        wins = sum(1 for s in sessions if s.state == "WIN")
        actions_to_win = [s.total_actions for s in sessions if s.state == "WIN"]
        times_to_win = [s.total_time for s in sessions if s.state == "WIN" and s.total_time > 0]
        unique_players = len(set(
            (s.player_name or s.user_id or "anon") for s in sessions
        ))
        
        daily = {}
        for s in sessions:
            day = s.started_at.strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"date": day, "plays": 0, "wins": 0, "times": []}
            daily[day]["plays"] += 1
            if s.state == "WIN":
                daily[day]["wins"] += 1
            if s.total_time > 0:
                daily[day]["times"].append(s.total_time)
        
        daily_stats = []
        for day_data in sorted(daily.values(), key=lambda x: x["date"]):
            daily_stats.append({
                "date": day_data["date"],
                "plays": day_data["plays"],
                "wins": day_data["wins"],
                "avg_time": round(sum(day_data["times"]) / len(day_data["times"]), 1) if day_data["times"] else 0,
            })
        
        return {
            "game_id": game.game_id,
            "game_name": game.name,
            "total_plays": total,
            "total_wins": wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "avg_actions_to_win": round(sum(actions_to_win) / len(actions_to_win), 1) if actions_to_win else 0,
            "avg_time_to_win": round(sum(times_to_win) / len(times_to_win), 1) if times_to_win else 0,
            "unique_players": unique_players,
            "daily_stats": daily_stats,
        }
    
    @staticmethod
    def get_dashboard_stats(db: Session) -> dict:
        total_games = db.query(Game).count()
        active_games = db.query(Game).filter(Game.is_active == True).count()
        total_plays = db.query(PlaySession).count()
        total_users = db.query(User).count()
        total_wins = db.query(PlaySession).filter(PlaySession.state == "WIN").count()
        total_game_overs = db.query(PlaySession).filter(PlaySession.state == "GAME_OVER").count()
        pending_requests = db.query(GameRequest).filter(GameRequest.status == "pending").count()

        recent_games = (
            db.query(Game)
            .order_by(Game.created_at.desc())
            .limit(5)
            .all()
        )

        top_played = (
            db.query(Game)
            .order_by(Game.total_plays.desc())
            .limit(5)
            .all()
        )

        # Daily play counts for last 7 days
        from app.models import now_ist
        from datetime import timedelta
        today = now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_plays = []
        daily_wins = []
        daily_labels = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            next_day = day + timedelta(days=1)
            plays = db.query(PlaySession).filter(
                PlaySession.started_at >= day,
                PlaySession.started_at < next_day,
            ).count()
            wins = db.query(PlaySession).filter(
                PlaySession.started_at >= day,
                PlaySession.started_at < next_day,
                PlaySession.state == "WIN",
            ).count()
            daily_plays.append(plays)
            daily_wins.append(wins)
            daily_labels.append(day.strftime("%b %d"))

        # Per-game play distribution
        game_distribution = []
        for g in db.query(Game).filter(Game.total_plays > 0).order_by(Game.total_plays.desc()).limit(8).all():
            game_distribution.append({
                "name": g.name or g.game_id,
                "plays": g.total_plays or 0,
                "wins": g.total_wins or 0,
            })

        # Win rate
        win_rate = round((total_wins / total_plays * 100), 1) if total_plays > 0 else 0

        return {
            "total_games": total_games,
            "active_games": active_games,
            "total_plays": total_plays,
            "total_users": total_users,
            "total_wins": total_wins,
            "total_game_overs": total_game_overs,
            "pending_requests": pending_requests,
            "win_rate": win_rate,
            "daily_plays": daily_plays,
            "daily_wins": daily_wins,
            "daily_labels": daily_labels,
            "game_distribution": game_distribution,
            "recent_games": recent_games,
            "top_played_games": [
                {
                    "game_id": g.game_id,
                    "name": g.name,
                    "total_plays": g.total_plays,
                    "total_wins": g.total_wins,
                }
                for g in top_played
            ],
        }
    
    @staticmethod
    def get_replay(db: Session, session_id: str) -> dict | None:
        session = db.query(PlaySession).filter(PlaySession.id == session_id).first()
        if not session:
            return None
        
        game = db.query(Game).filter(Game.id == session.game_id).first()
        
        return {
            "session_id": session.id,
            "game_id": game.game_id if game else "unknown",
            "game_name": game.name if game else "Unknown",
            "seed": session.seed,
            "state": session.state,
            "total_actions": session.total_actions,
            "total_time": session.total_time,
            "level_stats": session.level_stats or [],
            "player_name": session.player_name or "Anonymous",
            "action_log": session.action_log or [],
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        }
    
    @staticmethod
    def get_recent_sessions(db: Session, game_db_id: str | None = None, limit: int = 20) -> list[dict]:
        query = db.query(PlaySession).order_by(PlaySession.started_at.desc())
        
        if game_db_id:
            query = query.filter(PlaySession.game_id == game_db_id)
        
        sessions = query.limit(limit).all()
        
        results = []
        for s in sessions:
            game = db.query(Game).filter(Game.id == s.game_id).first()
            user = db.query(User).filter(User.id == s.user_id).first() if s.user_id else None
            results.append({
                "id": s.id,
                "game_name": game.name if game else "Unknown",
                "game_id": game.game_id if game else "unknown",
                "player": s.player_name or (user.username if user else "Anonymous"),
                "state": s.state,
                "score": s.score or 0,
                "total_actions": s.total_actions,
                "total_time": s.total_time or 0,
                "game_overs": s.game_overs or 0,
                "level_stats": s.level_stats or [],
                "current_level": s.current_level,
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            })
        
        return results
