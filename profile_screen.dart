import 'dart:ui';
import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

class ProfileScreen extends StatefulWidget {
  final Map<String, dynamic>? userData;
  final List<Map<String, dynamic>>? jobs;

  const ProfileScreen({Key? key, this.userData, this.jobs}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late Map<String, dynamic> user;
  late List<Map<String, dynamic>> jobHistory;

  @override
  void initState() {
    super.initState();

    // Default dynamic data (can be replaced by backend/fetch)
    user = widget.userData ?? {
      "name": "New Creator",
      "avatar": null,
      "bio": "Your creative journey begins!",
      "plan": "Free",
      "credits": 100,
      "expiry": "N/A",
      "storage": "1.2 GB / 10 GB",
      "badges": [
        {"title": "First Render", "icon": Icons.star},
        {"title": "AI Creator", "icon": Icons.auto_awesome},
      ],
      "stats": {
        "videos": 8,
        "templates": 5,
        "voices": 3,
        "renderMinutes": 45,
      },
      "settings": {
        "language": "Auto",
        "notifications": true,
        "darkMode": true,
      },
    };

    jobHistory = widget.jobs ??
        [
          {
            "title": "Motivational Hindi Video",
            "quality": "1080p",
            "language": "Hindi",
            "date": "Oct 17, 2025",
          },
          {
            "title": "AI Avatar Test",
            "quality": "4K",
            "language": "English",
            "date": "Oct 15, 2025",
          },
        ];
  }

  void _toggleSetting(String key, bool value) {
    setState(() {
      user["settings"][key] = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    final stats = user["stats"];
    final badges = user["badges"] as List;
    final settings = user["settings"];
    final plan = user["plan"].toString();

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          _animatedBackground(),
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _header(plan),
                  const SizedBox(height: 12),
                  _statsSection(stats),
                  const SizedBox(height: 12),
                  _planCard(plan),
                  const SizedBox(height: 12),
                  _badgesSection(badges),
                  const SizedBox(height: 12),
                  _recentJobs(jobHistory),
                  const SizedBox(height: 12),
                  _settingsSection(settings),
                  const SizedBox(height: 20),
                  _logoutButton(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _animatedBackground() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF833AB4), Color(0xFFFF5E62), Color(0xFFFCCF31)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
    );
  }

  Widget _header(String plan) {
    return Blur(
      blur: 6,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white12),
        ),
        child: Row(
          children: [
            CircleAvatar(
              radius: 40,
              backgroundImage: user["avatar"] != null
                  ? NetworkImage(user["avatar"])
                  : const AssetImage('assets/profile/default.png') as ImageProvider,
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(user["name"], style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w700)),
                  Text(user["bio"], style: const TextStyle(color: Colors.white70, fontSize: 12)),
                  const SizedBox(height: 8),
                  Row(children: [
                    Chip(
                      backgroundColor: plan == "Premium" ? Colors.purpleAccent : Colors.white24,
                      label: Text(plan, style: const TextStyle(color: Colors.white)),
                    ),
                    const SizedBox(width: 8),
                    const Icon(Icons.edit, color: Colors.white70, size: 18),
                  ]),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _statsSection(Map<String, dynamic> stats) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        _statCard("Videos", stats["videos"], Icons.movie_creation_outlined),
        _statCard("Templates", stats["templates"], Icons.widgets_outlined),
        _statCard("Voices", stats["voices"], Icons.record_voice_over_outlined),
        _statCard("Minutes", stats["renderMinutes"], Icons.timer_outlined),
      ],
    );
  }

  Widget _statCard(String label, int value, IconData icon) {
    return Expanded(
      child: Blur(
        blur: 4,
        borderRadius: BorderRadius.circular(14),
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 4),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.06),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(
            children: [
              Icon(icon, color: Colors.white70, size: 22),
              const SizedBox(height: 4),
              Text(value.toString(), style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
              Text(label, style: const TextStyle(fontSize: 10, color: Colors.white54)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _planCard(String plan) {
    return Blur(
      blur: 5,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.07),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text("Plan: $plan", style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
                Text("Credits: ${user["credits"]}", style: const TextStyle(color: Colors.white70, fontSize: 12)),
                Text("Expiry: ${user["expiry"]}", style: const TextStyle(color: Colors.white54, fontSize: 10)),
              ]),
            ),
            ElevatedButton(
              onPressed: () => _showSnack("Upgrade Plan"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurpleAccent,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
              child: const Text("Upgrade", style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _badgesSection(List badges) {
    return Blur(
      blur: 5,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Achievements", style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 10,
              children: badges.map((b) {
                return Chip(
                  backgroundColor: Colors.white10,
                  avatar: Icon(b["icon"], color: Colors.amber),
                  label: Text(b["title"], style: const TextStyle(color: Colors.white)),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _recentJobs(List<Map<String, dynamic>> jobs) {
    return Blur(
      blur: 5,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text("Recent Creations", style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ...jobs.map((j) => ListTile(
                leading: const Icon(Icons.video_library, color: Colors.white70),
                title: Text(j["title"], style: const TextStyle(color: Colors.white)),
                subtitle: Text("${j["language"]} • ${j["quality"]} • ${j["date"]}", style: const TextStyle(color: Colors.white54, fontSize: 12)),
                trailing: IconButton(
                  icon: const Icon(Icons.download, color: Colors.white70),
                  onPressed: () => _showSnack("Download ${j["title"]}"),
                ),
              )),
        ]),
      ),
    );
  }

  Widget _settingsSection(Map settings) {
    return Blur(
      blur: 5,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Settings", style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            SwitchListTile(
              value: settings["notifications"],
              onChanged: (v) => _toggleSetting("notifications", v),
              title: const Text("Notifications", style: TextStyle(color: Colors.white)),
            ),
            SwitchListTile(
              value: settings["darkMode"],
              onChanged: (v) => _toggleSetting("darkMode", v),
              title: const Text("Dark Mode", style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _logoutButton() {
    return ElevatedButton.icon(
      onPressed: () => _showSnack("Logout / Switch Account"),
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.redAccent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      icon: const Icon(Icons.logout),
      label: const Text("Logout", style: TextStyle(color: Colors.white)),
    );
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }
}
