import 'dart:async';
import 'dart:math';
import 'dart:ui';
import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

class TemplatesScreen extends StatefulWidget {
  const TemplatesScreen({Key? key}) : super(key: key);

  @override
  State<TemplatesScreen> createState() => _TemplatesScreenState();
}

class _TemplatesScreenState extends State<TemplatesScreen> with TickerProviderStateMixin {
  bool _loading = true;
  String _query = "";
  String _selectedCategory = "All";

  List<Map<String, dynamic>> _templates = [];
  List<String> _categories = [
    "All",
    "Motivation",
    "Business",
    "Education",
    "Meme",
    "Festival",
    "Health",
    "Travel",
    "Gaming"
  ];

  late AnimationController _bgController;
  late Animation<double> _bgAnim;

  @override
  void initState() {
    super.initState();
    _bgController = AnimationController(vsync: this, duration: const Duration(seconds: 12))..repeat(reverse: true);
    _bgAnim = Tween<double>(begin: -0.3, end: 0.3).animate(CurvedAnimation(parent: _bgController, curve: Curves.easeInOut));
    _fetchTemplates();
  }

  Future<void> _fetchTemplates() async {
    setState(() => _loading = true);
    await Future.delayed(const Duration(seconds: 1));
    final rnd = Random();

    _templates = List.generate(12, (i) {
      final cats = ["Motivation", "Business", "Education", "Meme", "Festival", "Health", "Travel"];
      final cat = cats[rnd.nextInt(cats.length)];
      return {
        "id": "t$i",
        "title": "$cat Template #$i",
        "category": cat,
        "aspect": rnd.nextBool() ? "9:16" : "16:9",
        "duration": "${20 + rnd.nextInt(40)}s",
        "tags": ["AI", cat],
        "preview": "https://example.com/previews/temp$i.mp4",
        "thumbnail": "https://picsum.photos/seed/$i/400/600",
      };
    });
    setState(() => _loading = false);
  }

  List<Map<String, dynamic>> get _filteredTemplates {
    return _templates.where((t) {
      final matchCategory = _selectedCategory == "All" || t["category"] == _selectedCategory;
      final matchQuery = _query.isEmpty || t["title"].toString().toLowerCase().contains(_query.toLowerCase());
      return matchCategory && matchQuery;
    }).toList();
  }

  @override
  void dispose() {
    _bgController.dispose();
    super.dispose();
  }

  Widget _animatedBackground() {
    return AnimatedBuilder(
      animation: _bgAnim,
      builder: (context, child) {
        final v = _bgAnim.value;
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(-1 + v, -0.8 - v),
              end: Alignment(1 - v, 0.8 + v),
              colors: [Color(0xFF833AB4), Color(0xFFFF5E62), Color(0xFFFCCF31)],
            ),
          ),
        );
      },
    );
  }

  Widget _header() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
        child: Row(
          children: [
            Text("Templates", style: GoogleFonts.poppins(fontSize: 22, fontWeight: FontWeight.w700)),
            const Spacer(),
            IconButton(onPressed: _fetchTemplates, icon: const Icon(Icons.refresh, color: Colors.white)),
          ],
        ),
      ),
    );
  }

  Widget _searchBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
      child: TextField(
        onChanged: (v) => setState(() => _query = v),
        decoration: InputDecoration(
          hintText: "Search templates...",
          prefixIcon: const Icon(Icons.search),
          filled: true,
          fillColor: Colors.white10,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
        ),
        style: const TextStyle(color: Colors.white),
      ),
    );
  }

  Widget _categoryChips() {
    return SizedBox(
      height: 40,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemBuilder: (_, i) {
          final cat = _categories[i];
          final selected = _selectedCategory == cat;
          return ChoiceChip(
            label: Text(cat),
            selected: selected,
            onSelected: (_) => setState(() => _selectedCategory = cat),
            selectedColor: Colors.purpleAccent,
            backgroundColor: Colors.white10,
            labelStyle: const TextStyle(color: Colors.white),
          );
        },
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemCount: _categories.length,
      ),
    );
  }

  Widget _templateGrid() {
    final filtered = _filteredTemplates;
    if (_loading) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Lottie.asset('assets/lottie/loading.json', width: 140),
            const Text("Loading templates...", style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }
    if (filtered.isEmpty) {
      return const Center(child: Text("No templates found", style: TextStyle(color: Colors.white70)));
    }
    return GridView.builder(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.all(16),
      itemCount: filtered.length,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 9 / 16,
      ),
      itemBuilder: (context, i) {
        final t = filtered[i];
        return GestureDetector(
          onTap: () => _openPreview(t),
          child: Blur(
            blur: 6,
            borderRadius: BorderRadius.circular(16),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(16),
                image: DecorationImage(
                  image: NetworkImage(t["thumbnail"]),
                  fit: BoxFit.cover,
                  colorFilter: ColorFilter.mode(Colors.black26, BlendMode.darken),
                ),
              ),
              child: Stack(
                children: [
                  Positioned(
                    left: 8,
                    top: 8,
                    child: Chip(
                      backgroundColor: Colors.white12,
                      label: Text(t["category"], style: const TextStyle(color: Colors.white, fontSize: 10)),
                    ),
                  ),
                  Positioned(
                    bottom: 10,
                    left: 8,
                    right: 8,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(t["title"], maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                        Text("${t["duration"]} • ${t["aspect"]}", style: const TextStyle(color: Colors.white70, fontSize: 12)),
                      ],
                    ),
                  )
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Future<void> _openPreview(Map<String, dynamic> template) async {
    await showModalBottomSheet(
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      context: context,
      builder: (_) => _PreviewSheet(template: template),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          _animatedBackground(),
          Column(
            children: [
              _header(),
              _searchBar(),
              _categoryChips(),
              Expanded(child: _templateGrid()),
            ],
          ),
        ],
      ),
    );
  }
}

class _PreviewSheet extends StatelessWidget {
  final Map<String, dynamic> template;
  const _PreviewSheet({required this.template});

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.75,
      maxChildSize: 0.95,
      minChildSize: 0.5,
      builder: (context, scroll) => Blur(
        blur: 10,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
        child: Container(
          decoration: const BoxDecoration(
            color: Color(0xFF0F0F12),
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          padding: const EdgeInsets.all(16),
          child: ListView(
            controller: scroll,
            children: [
              Center(
                child: Container(width: 60, height: 6, decoration: BoxDecoration(color: Colors.white12, borderRadius: BorderRadius.circular(4))),
              ),
              const SizedBox(height: 12),
              Text(template["title"], style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w700)),
              const SizedBox(height: 8),
              Text("${template["category"]} • ${template["duration"]} • ${template["aspect"]}", style: const TextStyle(color: Colors.white70)),
              const SizedBox(height: 16),
              Container(
                height: 200,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(12),
                  image: DecorationImage(image: NetworkImage(template["thumbnail"]), fit: BoxFit.cover),
                ),
                child: const Center(child: Icon(Icons.play_circle_outline, color: Colors.white, size: 60)),
              ),
              const SizedBox(height: 16),
              Text("Tags: ${template["tags"].join(", ")}", style: const TextStyle(color: Colors.white70)),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text("Template '${template["title"]}' applied")),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurpleAccent,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                icon: const Icon(Icons.check_circle_outline),
                label: const Text("Use Template", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
